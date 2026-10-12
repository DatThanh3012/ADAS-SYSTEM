from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import torch
import os
import shutil
from ultralytics import YOLO
import segmentation_models_pytorch as smp

app = FastAPI()

# --- 1. CẤU HÌNH CORS (Bắt buộc để Frontend React chạy được) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục lưu video kết quả
RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 2. LOAD MODELS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO("yolov8n.pt") 
seg_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, activation='sigmoid').to(device)
seg_model.load_state_dict(torch.load("unet.pth", map_location=device))
seg_model.eval()

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    # Lưu video tạm thời
    temp_input = f"temp_{video.filename}"
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(temp_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_filename = f"processed_{video.filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)
    
    # Dùng codec 'mp4v' để web có thể đọc được
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stats = {"traffic_sign": 0, "car": 0, "motorbike": 0, "person": 0}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- 0. KHỞI TẠO VÙNG NGUY HIỂM (ROI) TRƯỚC ---
        # Định nghĩa ROI ở đây để dùng cho cả YOLO và UNET
        # --- TÍNH TOÁN VÙNG NGUY HIỂM THÍCH NGHI (ADAPTIVE ROI) ---
        is_portrait = height > width
        
        if is_portrait:
            # Video dọc
            roi_x1 = int(width * 0.25) 
            roi_x2 = int(width * 0.75)
            roi_y1 = int(height * 0.50)
            roi_y2 = int(height * 0.85) 
        else:
            # Video ngang
            roi_x1 = int(width * 0.40)
            roi_x2 = int(width * 0.60)
            roi_y1 = int(height * 0.65) 
            roi_y2 = int(height * 0.90)
            
        frame_area = width * height
        current_status = "SAFE"

        # --- 1. CHẠY YOLO ---
        yolo_res = yolo_model(frame, verbose=False)[0]
        
        #logic va chạm
        for box in yolo_res.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Cập nhật nhãn 
            if cls == 0: label = "traffic_sign"; stats["traffic_sign"] += 1
            elif cls == 1: label = "car"; stats["car"] += 1
            elif cls == 2: label = "motorbike"; stats["motorbike"] += 1
            elif cls == 3: label = "person"; stats["person"] += 1
            else: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) / 2
            box_area = (x2 - x1) * (y2 - y1)

            # --- KIỂM TRA VA CHẠM (COLLISION) ---
            # Nếu là xe/người VÀ to (gần) VÀ nằm ở giữa làn
            if cls in [1, 2, 3] and (box_area / frame_area > 0.05) and (roi_x1 < center_x < roi_x2):
                current_status = "COLLISION"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4) # Khung đỏ đậm
                cv2.putText(frame, "WARNING!", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # Vẽ khung xanh bình thường nếu không nguy hiểm
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --- 2. CHẠY UNET (LANE) ---
        img_input = cv2.resize(frame, (256, 256)) / 255.0
        img_tensor = torch.tensor(img_input).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            mask = seg_model(img_tensor)[0][0].cpu().numpy()
        
        mask_resized = cv2.resize(mask, (width, height))
        lane_color = np.zeros_like(frame)
        lane_color[mask_resized > 0.5] = [0, 255, 0] 
        frame = cv2.addWeighted(frame, 1, lane_color, 0.4, 0)

        # --- 3. KIỂM TRA LỆCH LÀN (LANE DEPARTURE) ---
        if current_status == "SAFE":
            ego_mask = mask_resized[roi_y1:roi_y2, roi_x1:roi_x2]
            
            lane_pixels = np.sum(ego_mask > 0.5)
            total_roi_pixels = (roi_y2 - roi_y1) * (roi_x2 - roi_x1)
            lane_ratio = lane_pixels / total_roi_pixels

            if lane_ratio > 0.01: 
                current_status = "LANE_DEPARTURE"
                cv2.putText(frame, "LANE DEPARTURE", 
                            (int(width/2) - 180, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 3)

        # Hiển thị vùng Ego-zone mờ mờ để debug
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)
        adas_status = current_status
        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_input)

    return {
        "video_url": f"/static/results/{output_filename}",
        "stats": { "traffic_sign": stats["traffic_sign"], "car": stats["car"], "motorbike": stats["motorbike"], "person": stats["person"],  },
        "adas_status": adas_status # "SAFE", "COLLISION", hoặc "LANE_DEPARTURE"
    }