import torch
import torchmetrics


def get_auc(preds, masks):
    """
    Calculate the area under the curve (AUC) for the given predictions and masks.

    Args:
        preds (torch.Tensor): The predictions.
            preds.shape: (batch_size, height, width)
        masks (torch.Tensor): The masks.
            masks.shape: (batch_size, height, width)

    Returns:
        float: The AUC.
    """
    preds_flat = torch.argmax(preds, dim=1).flatten()
    masks_flat = torch.argmax(masks, dim=1).flatten()
    absolute_auc = torch.eq(preds_flat, masks_flat).float().mean().item()
    return absolute_auc
