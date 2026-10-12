import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# ===== Dataset =====
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = os.listdir(img_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img = img / 255.0
        mask = mask / 255.0
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


# ===== U-Net =====
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )
        self.enc1 = block(3, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = block(64, 128)
        self.middle = block(128, 256)
        self.up = nn.Upsample(scale_factor=2)
        self.dec1 = block(256, 128)
        self.dec2 = block(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.middle(self.pool(e2))
        d1 = self.up(m)
        d1 = self.dec1(d1)
        d2 = self.up(d1)
        d2 = self.dec2(d2)
        return torch.sigmoid(self.out(d2))


# ===== KHỐI LỆNH THỰC THI =====
if __name__ == "__main__":
    print("Bắt đầu huấn luyện...")
    
    # Khởi tạo data
    dataset = SegDataset("dataset_final/seg/images", "dataset_final/seg/masks")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Khởi tạo model
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Vòng lặp huấn luyện
    for epoch in range(10):
        total_loss = 0
        for img, mask in loader:
            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Lưu model
    torch.save(model.state_dict(), "unet.pth")
    print("Đã huấn luyện xong và lưu model thành công!")