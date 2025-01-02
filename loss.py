import torch
import torch.nn.functional as F

# mIoU Loss
def mIoU_loss(P, G, num_classes):
    """
    Mean Intersection over Union (mIoU) loss
    Args:
        P: predicted segmentation (Tensor of shape [batch_size, num_classes, H, W])
        G: ground truth segmentation (Tensor of shape [batch_size, num_classes, H, W])
        num_classes: number of classes in the segmentation task
    Returns:
        mIoU loss value
    """
    intersection = torch.sum(P * G, dim=(2, 3))  # Calculate intersection (True Positives)
    union = torch.sum(P, dim=(2, 3)) + torch.sum(G, dim=(2, 3)) - intersection  # Calculate union (True Positives + False Positives + False Negatives)
    
    # Compute the IoU for each class and average over classes
    iou = (intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
    mIoU = 1 - torch.mean(iou)  # mIoU loss (we subtract from 1 to minimize the loss)
    
    return mIoU

# Dice Loss
def dice_loss(P, G):
    """
    Dice loss function
    Args:
        P: predicted segmentation (Tensor of shape [batch_size, num_classes, H, W])
        G: ground truth segmentation (Tensor of shape [batch_size, num_classes, H, W])
    Returns:
        Dice loss value
    """
    smooth = 1e-6  # Smoothing constant to avoid division by zero
    intersection = torch.sum(P * G, dim=(2, 3))  # Calculate intersection (True Positives)
    
    # Calculate Dice Coefficient
    dice = (2.0 * intersection + smooth) / (torch.sum(P, dim=(2, 3)) + torch.sum(G, dim=(2, 3)) + smooth)
    dice_loss_value = 1 - torch.mean(dice)  # Dice loss (we subtract from 1 to minimize the loss)
    
    return dice_loss_value

# Boundary Loss
def boundary_loss(P, G, threshold=0.5):
    """
    Boundary loss function based on distance between predicted and ground truth boundaries
    Args:
        P: predicted segmentation (Tensor of shape [batch_size, num_classes, H, W])
        G: ground truth segmentation (Tensor of shape [batch_size, num_classes, H, W])
        threshold: threshold to define the boundaries
    Returns:
        Boundary loss value
    """
    # Convert to binary masks
    P_bin = (P > threshold).float()
    G_bin = (G > threshold).float()
    
    # Compute the boundaries using morphological operations (using gradient)
    P_boundary = F.max_pool2d(P_bin, kernel_size=3, stride=1, padding=1) - P_bin
    G_boundary = F.max_pool2d(G_bin, kernel_size=3, stride=1, padding=1) - G_bin
    
    # Calculate the distance between predicted and ground truth boundaries
    boundary_distance = torch.sum(torch.abs(P_boundary - G_boundary), dim=(2, 3))  # Calculate the boundary distance
    
    # Normalize by the number of pixels in the boundary region
    boundary_loss_value = torch.mean(boundary_distance)
    
    return boundary_loss_value

# Total Loss (Weighted combination of mIoU, Dice, and Boundary Loss)
def total_loss(P, G, alpha=1.0, beta=1.0, gamma=1.0, num_classes=2):
    """
    Weighted total loss function combining mIoU, Dice, and Boundary losses
    Args:
        P: predicted segmentation (Tensor of shape [batch_size, num_classes, H, W])
        G: ground truth segmentation (Tensor of shape [batch_size, num_classes, H, W])
        alpha, beta, gamma: weights for mIoU, Dice, and Boundary loss respectively
        num_classes: number of classes in the segmentation task
    Returns:
        Total loss value
    """
    mIoU = mIoU_loss(P, G, num_classes)
    dice = dice_loss(P, G)
    boundary = boundary_loss(P, G)
    
    # Combine the losses using weighted sum
    total_loss_value = alpha * mIoU + beta * dice + gamma * boundary
    
    return total_loss_value
