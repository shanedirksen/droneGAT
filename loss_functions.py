import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_losses(outputs, targets, args):
    """
    Compute losses for model outputs against targets
    
    Args:
        outputs: Model outputs
        targets: Target values
        args: Command line arguments
        
    Returns:
        total_loss, loss_components
    """
    loss_components = {}
    total_loss = 0.0
    
    # Depth prediction loss
    if 'depth' in outputs and 'depth' in targets:
        # Use L1 loss for depth
        depth_loss = nn.L1Loss()(outputs['depth'], targets['depth'])
        # print("depth loss", depth_loss)
        total_loss += depth_loss
        loss_components['depth'] = depth_loss.item()
    
    # Segmentation prediction loss
    if 'segmentation' in outputs and 'segmentation' in targets:
        # For segmentation, we need to convert the color mask to class indices
        # if the targets are in color format
        target_seg = targets['segmentation']
        
        # Check if target is a color mask (3-channel) and convert if needed
        if target_seg.dim() == 4 and target_seg.size(1) == 3:  # [B, 3, H, W]
            # Note: This is a simplified approach. For production, replace with a proper
            # color-to-index mapping function based on your specific dataset
            
            # Simple approach: use argmax across color channels as class index
            # This assumes the brightest channel represents the class
            target_seg = target_seg.argmax(dim=1)  # [B, H, W]
        
        # If target has shape [B, 1, H, W], squeeze out the channel dimension to get [B, H, W]
        elif target_seg.dim() == 4 and target_seg.size(1) == 1:
            target_seg = target_seg.squeeze(1)
        
        # Convert target to long type for CrossEntropyLoss
        seg_loss = nn.CrossEntropyLoss()(outputs['segmentation'], target_seg.long())
        
        # Apply a weight of 1000 to make segmentation loss stronger
        seg_weight = 1000.0
        weighted_seg_loss = seg_loss * seg_weight
        
        # print("segmentation loss", seg_loss)
        total_loss += weighted_seg_loss
        loss_components['segmentation'] = seg_loss.item()  # Store the original unweighted loss
        
    return total_loss, loss_components
