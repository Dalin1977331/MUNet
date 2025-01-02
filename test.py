import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MUNet  # 导入训练好的 MUNet 模型
from dataset import CustomDataset  # 自定义数据集类
import os
import numpy as np
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 参数设置
TEST_DATASET_PATH = "path/to/test/dataset"
CHECKPOINT_PATH = "./checkpoints/checkpoint_epoch_best.pth"  # 使用保存的最佳权重
OUTPUT_DIR = "./outputs"  # 保存测试结果
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 256
IN_CHANNELS = 1
OUT_CHANNELS = 1
EMBED_DIM = 64
NUM_LAYERS = 4

# 数据增强（与训练时一致）
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# 加载测试数据集
test_dataset = CustomDataset(TEST_DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 加载模型
model = MUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)  # 加载检查点
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 设置为评估模式
print(f"Loaded model from {CHECKPOINT_PATH}")

# 推理函数
def predict(loader, model, output_dir):
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(loader):  # 假设自定义数据集返回图像和文件名
            images = images.to(device)

            # 前向传播
            preds = model(images)
            preds = torch.sigmoid(preds)  # 使用 Sigmoid 将输出归一化到 [0, 1]
            preds = (preds > 0.5).float()  # 二值化分割结果

            # 保存结果
            for i in range(preds.size(0)):
                pred_mask = preds[i].cpu().numpy().squeeze()  # 转换为 NumPy 格式
                save_path = os.path.join(output_dir, filenames[i])
                save_mask(pred_mask, save_path)

# 保存分割结果为图像文件
def save_mask(mask, save_path):
    """
    将分割结果保存为图像文件
    Args:
        mask: NumPy 格式的二值化分割结果
        save_path: 保存路径
    """
    mask = (mask * 255).astype(np.uint8)  # 将 mask 转换为 0-255 范围
    mask_image = Image.fromarray(mask)  # 转换为 PIL 图像
    mask_image.save(save_path)  # 保存图像
    print(f"Saved: {save_path}")

# 测试函数
def test():
    print("Starting testing...")
    predict(test_loader, model, OUTPUT_DIR)
    print("Testing completed. Results saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    test()
