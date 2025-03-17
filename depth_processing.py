import torch
import torch.nn as nn
import numpy as np

# Import the depth processing function from depth_visualization
from depth_visualization import process_single_depth_map

def process_depth_maps(depth_tensor):
    """
    Process depth maps in PyTorch tensor format with same rules as depth_visualization.py
    
    Args:
        depth_tensor: PyTorch tensor with depth maps [B, C, H, W] or [B, H, W]
        
    Returns:
        Processed depth tensor
    """
    # No need to convert to PIL or normalize - work directly with the float values
    # Convert to numpy for processing
    if depth_tensor.dim() == 4:  # [B, C, H, W]
        # Process each batch and channel
        processed_depth = []
        for batch_idx in range(depth_tensor.shape[0]):
            processed_channels = []
            for channel_idx in range(depth_tensor.shape[1]):
                # Get the depth map for this batch and channel
                depth_map = depth_tensor[batch_idx, channel_idx].cpu().numpy()
                # Process using the visualization function
                processed_depth_map = process_single_depth_map(depth_map)
                processed_channels.append(torch.from_numpy(processed_depth_map).to(depth_tensor.device))
            processed_depth.append(torch.stack(processed_channels, dim=0))
        return torch.stack(processed_depth, dim=0)
    elif depth_tensor.dim() == 3:  # [B, H, W]
        # Process each batch
        processed_depth = []
        for batch_idx in range(depth_tensor.shape[0]):
            depth_map = depth_tensor[batch_idx].cpu().numpy()
            processed_depth_map = process_single_depth_map(depth_map)
            processed_depth.append(torch.from_numpy(processed_depth_map).to(depth_tensor.device))
        return torch.stack(processed_depth, dim=0)
    else:
        # Single depth map [H, W]
        depth_map = depth_tensor.cpu().numpy()
        processed_depth_map = process_single_depth_map(depth_map)
        return torch.from_numpy(processed_depth_map).to(depth_tensor.device)

def compute_losses_with_processed_depth(outputs, targets, args):
    """
    Updated compute_losses function that knows depth maps have already been processed
    
    Args:
        outputs: Model outputs
        targets: Target values (depth maps already processed)
        args: Command line arguments
        
    Returns:
        total_loss, loss_components
    """
    loss_components = {}
    
    # Select L1 loss for depth prediction
    depth_criterion = nn.L1Loss()
    
    # For segmentation, use cross-entropy loss
    seg_criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    
    # Depth prediction loss
    if 'depth' in outputs and 'depth' in targets:
        depth_loss = 0.0
        num_valid_depths = 0
        
        # Process each item in the batch
        for i, target_depth in enumerate(targets['depth']):
            if i < len(outputs['depth']):
                # No need for additional processing since the depth maps were processed
                # during the prepare_batch_for_model step
                pred_depth = outputs['depth'][i]
                curr_depth_loss = depth_criterion(pred_depth, target_depth)
                depth_loss += curr_depth_loss
                num_valid_depths += 1
        
        if num_valid_depths > 0:
            depth_loss /= num_valid_depths
            total_loss += depth_loss
            loss_components['depth'] = depth_loss.item()
    
    # Segmentation prediction loss
    if 'segmentation' in outputs and 'segmentation' in targets:
        seg_loss = 0.0
        num_valid_segs = 0
        
        # Process each item in the batch
        for i, target_seg in enumerate(targets['segmentation']):
            if i < len(outputs['segmentation']):
                pred_seg = outputs['segmentation'][i]
                
                # Remove the channel dimension from target if it exists
                if target_seg.dim() == 4 and target_seg.size(1) == 1:  # [B, 1, H, W]
                    target_seg = target_seg.squeeze(1)  # Convert to [B, H, W]
                
                # Reshape for cross-entropy loss
                b, c, h, w = pred_seg.size()
                pred_seg_flat = pred_seg.permute(0, 2, 3, 1).reshape(-1, c)
                target_seg_flat = target_seg.reshape(-1).long()  # Ensure target indices are long type
                
                curr_seg_loss = seg_criterion(pred_seg_flat, target_seg_flat)
                seg_loss += curr_seg_loss
                num_valid_segs += 1
        
        if num_valid_segs > 0:
            seg_loss /= num_valid_segs
            total_loss += seg_loss
            loss_components['segmentation'] = seg_loss.item()
    
    return total_loss, loss_components
