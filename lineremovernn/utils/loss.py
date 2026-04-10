import torch
import torch.nn as nn


class CombinedDiceBCEWithLogitsLoss(nn.Module):
    """
    Combines Dice Loss and Binary Cross-Entropy (BCE) for stable segmentation training.
    Assumes model outputs logits (no sigmoid applied by the model).
    """

    def __init__(self, smooth=1e-6, weight_dice=0.5, weight_bce=0.5):
        super(CombinedDiceBCEWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

        # BCEWithLogitsLoss is the preferred function as it combines sigmoid and BCE loss.
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        """Calculates the Dice Loss component."""
        # pred and target must be logits and probabilities, respectively.
        # Since we use BCEWithLogitsLoss, we must calculate Dice on probabilities (after sigmoid).
        pred_prob = torch.sigmoid(pred)

        # Flatten the spatial dimensions (Batch * Channel * H * W)
        pred_s = pred_prob.view(-1)
        target_s = target.view(-1)

        intersection = (pred_s * target_s).sum()
        union = pred_s.sum() + target_s.sum()

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coefficient

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Model logits (N, 1, H, W).
            target (torch.Tensor): Ground truth mask (N, 1, H, W) with float type (0.0 or 1.0).
        Returns:
            torch.Tensor: Combined loss value.
        """
        # 1. Calculate Dice Loss
        loss_dice = self.dice_loss(pred, target)

        # 2. Calculate BCE Loss (using logits directly)
        loss_bce = self.bce_loss(pred, target)

        # 3. Combine losses
        total_loss = (self.weight_dice * loss_dice) + (self.weight_bce * loss_bce)
        return total_loss
