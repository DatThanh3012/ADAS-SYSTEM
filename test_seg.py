import cv2
import torch
import numpy as np
import os
import segmentation_models_pytorch as smp

def run_test():
    # --- 1. Cấu hình ---
    model_path = "unet.pth"
    img_dir = "dataset_raw/test/images"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy {model_path}"); return

    # --- 2. Load Model ---
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Đã load model thành công!")

    # --- 3. Lấy ảnh test ---
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print("Thư mục trống!"); return
    
    test_img_path = os.path.join(img_dir, images[9])
    img_raw = cv2.imread(test_img_path)

    # Pre-processing (Đưa về 256x256 để model dự đoán)
    img_input = cv2.resize(img_raw, (256, 256))
    img_input = img_input / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).to(device)

    # --- 4. Dự đoán ---
    with torch.no_grad():
        pred = model(img_input)[0][0].cpu().numpy()

    # --- 5. Xử lý hiển thị ---
    # Chuyển kết quả thành Mask đen trắng 0-255
    mask_binary = (pred > 0.5).astype(np.uint8) * 255

    # --- 6. Show kết quả (Cố định kích thước 256x256 để hình nhỏ vừa phải) ---
    print(f"Giá trị dự đoán cao nhất: {pred.max():.4f}")
    
    cv2.imshow("1. Anh Goc", cv2.resize(img_raw, (256, 256)))
    cv2.imshow("2. Mask", mask_binary) # mask_binary vốn đã là 256x256 từ đầu
    
    print("Nhấn phím bất kỳ để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()