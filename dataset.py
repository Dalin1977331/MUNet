import os
import glob
import nibabel as nib  # 用于加载 NIfTI 格式文件
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    """
    BraTS2020 数据集，用于加载多模态 MRI 图像 (Flair, T1, T1ce, T2) 和分割标签。
    """
    def __init__(self, root_dir, num_patients=None, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录，包含各病例的子文件夹。
            num_patients (int, optional): 加载的患者数量（默认加载全部）。
            transform (callable, optional): 数据增强转换函数（默认无）。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有患者文件夹路径
        self.patients = sorted(os.listdir(root_dir))
        if num_patients:
            self.patients = self.patients[:num_patients]  # 限制加载的患者数量

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        """
        获取指定索引的 MRI 数据和分割标签
        Args:
            idx (int): 数据索引
        Returns:
            image (Tensor): 合并后的多模态 MRI 图像 (Flair, T1, T1ce, T2)
            seg (Tensor): 分割标签
        """
        patient_id = self.patients[idx]
        patient_path = os.path.join(self.root_dir, patient_id)

        # 加载各模态文件路径
        flair_path = glob.glob(os.path.join(patient_path, '*_flair*.nii'))[0]
        t1_path = glob.glob(os.path.join(patient_path, '*_t1*.nii'))[0]
        t1ce_path = glob.glob(os.path.join(patient_path, '*_t1ce*.nii'))[0]
        t2_path = glob.glob(os.path.join(patient_path, '*_t2*.nii'))[0]
        seg_path = glob.glob(os.path.join(patient_path, '*_seg*.nii'))[0]

        # 预处理数据
        image, seg = self.data_preprocessing([flair_path, t1_path, t1ce_path, t2_path], seg_path)

        # 数据增强（可选）
        if self.transform:
            image, seg = self.transform((image, seg))

        # 转换为 PyTorch 张量
        image = torch.tensor(image, dtype=torch.float32)  # 图像数据为 float32
        seg = torch.tensor(seg, dtype=torch.long)         # 标签数据为 long 类型

        return image, seg

    def data_preprocessing(self, modalities_dir, seg_dir):
        """
        加载和预处理 MRI 数据和分割标签
        Args:
            modalities_dir (list): 多模态 MRI 数据的文件路径列表
            seg_dir (str): 分割标签的文件路径
        Returns:
            all_modalities (numpy array): 合并后的多模态 MRI 数据 (C, H, W, D)
            seg (numpy array): 分割标签 (H, W, D)
        """
        all_modalities = []
        for modality in modalities_dir:
            nifti_file = nib.load(modality)
            brain_numpy = np.asarray(nifti_file.dataobj)  # 转为 NumPy 数组
            all_modalities.append(brain_numpy)

        # 加载分割标签
        seg_nifti = nib.load(seg_dir)
        seg = np.asarray(seg_nifti.dataobj)

        # 转置维度以匹配 PyTorch 格式 (C, H, W, D)
        all_modalities = np.array(all_modalities)  # 转为 NumPy 数组
        all_modalities = np.transpose(all_modalities, (0, 1, 2, 3))

        return all_modalities, seg


# 自定义数据增强
class RandomFlip:
    """
    随机翻转图像和标签，用于数据增强
    """
    def __call__(self, sample):
        image, seg = sample
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2)  # 随机水平翻转
            seg = np.flip(seg, axis=1)
        return image, seg


# 测试数据加载
if __name__ == "__main__":
    # 数据集路径
    dataset_path = '../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

    # 初始化数据集
    transform = RandomFlip()  # 使用自定义数据增强
    dataset = BraTSDataset(root_dir=dataset_path, num_patients=10, transform=transform)

    # 创建 DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # 测试数据加载
    for images, segs in data_loader:
        print(f"Images shape: {images.shape}")  # (batch_size, 4, H, W, D)
        print(f"Segs shape: {segs.shape}")      # (batch_size, H, W, D)
        break
