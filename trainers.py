import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from loss_functions import compute_losses
from utils import (
    prepare_batch_for_model,
    compute_depth_metrics,
    compute_segmentation_metrics
)

def train_epoch(model, data_loader, optimizer, device, args):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0.0
    loss_components_sum = {}
    batch_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        # Skip empty batches
        if not batch or 'rgb' not in batch:
            continue
        
        # Prepare data for model (now using the function from utils)
        rgb_features, edge_indices, rel_poses, targets = prepare_batch_for_model(batch, device, args.distance_threshold)
        
        total_batch_loss = 0.0
        batch_components = {
            'depth': 0.0, 
            'depth_l1': 0.0, 
            'edge_smoothness': 0.0, 
            'segmentation': 0.0, 
            'features': 0.0
        }
        batch_count = 0
        
        # Process each item in the batch
        for j, (features, edge_index, rel_pose) in enumerate(zip(rgb_features, edge_indices, rel_poses)):
            # Skip if no edges
            if edge_index.size(1) == 0:
                continue
                
            # Debug print for very first batch of first epoch only
            if args.verbose and batch_idx == 0 and j == 0 and batch_count == 0:
                print("\nDebug information (first batch only):")
                print(f"  Features device: {features.device}")
                print(f"  Edge index device: {edge_index.device}")
                print(f"  Features shape: {features.shape}")
                print(f"  Model device: {next(model.parameters()).device}")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, edge_index, rel_pose)
            
            # Debug print for first batch
            if args.verbose and batch_idx == 0 and j == 0 and batch_count == 0:
                print(f"  Output keys: {outputs.keys()}")
                for k, v in outputs.items():
                    print(f"  Output '{k}' shape: {v.shape}")
                print("")
            
            # Prepare targets for this item
            item_targets = {
                'rgb': features,  # Add RGB input for edge-aware smoothness loss
                'depth': targets['depth'][j] if 'depth' in targets and j < len(targets['depth']) else None,
                'segmentation': targets['segmentation'][j] if 'segmentation' in targets and j < len(targets['segmentation']) else None
            }
            
            # Compute loss
            loss, loss_components = compute_losses(outputs, item_targets, args)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_batch_loss += loss.item()
            batch_count += 1
            
            # Accumulate loss components
            for component, value in loss_components.items():
                batch_components[component] += value
        
        # Skip iteration if no valid inputs in batch
        if batch_count == 0:
            continue
        
        # Average loss for this batch
        batch_loss = total_batch_loss / batch_count
        total_loss += batch_loss
        batch_count += 1
        
        # Average loss components
        for component in loss_components_sum.keys():
            if component in batch_components and batch_components[component] > 0:
                loss_components_sum[component] += batch_components[component] / batch_count
        
        # Log training progress
        if args.verbose and batch_idx % args.log_interval == 0:
            component_str = ', '.join([f"{k}: {v/batch_count:.4f}" for k, v in batch_components.items() if v > 0])
            print(f"Batch {batch_idx:3d}/{len(data_loader)}, Loss: {batch_loss:.4f} ({component_str})")
    
    # Return average loss over the epoch
    if batch_count == 0:
        return 0.0, {}
    
    avg_components = {k: v/batch_count for k, v in loss_components_sum.items() if v > 0}
    return total_loss / batch_count, avg_components


def validate(model, dataloader, device, args):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0.0
    val_components = {
        'depth': 0.0, 
        'depth_l1': 0.0, 
        'edge_smoothness': 0.0, 
        'segmentation': 0.0, 
        'features': 0.0
    }
    
    # Metrics tracking
    depth_metrics = {
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'rmse': 0.0
    }
    
    seg_metrics = {
        'miou': 0.0,
        'pixel_acc': 0.0
    }
    
    num_batches = 0
    num_depth_samples = 0
    num_seg_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Skip empty batches
            if not batch or 'rgb' not in batch:
                continue
            
            # Prepare data for model (now using the function from utils)
            rgb_features, edge_indices, rel_poses, targets = prepare_batch_for_model(batch, device, args.distance_threshold)
            
            batch_loss = 0.0
            batch_components = {
                'depth': 0.0, 
                'depth_l1': 0.0, 
                'edge_smoothness': 0.0, 
                'segmentation': 0.0, 
                'features': 0.0
            }
            batch_count = 0
            
            # Process each item in the batch
            for j, (features, edge_index, rel_pose) in enumerate(zip(rgb_features, edge_indices, rel_poses)):
                # Skip if no edges
                if edge_index.size(1) == 0:
                    continue
                
                # Forward pass
                outputs = model(features, edge_index, rel_pose)
                
                # Prepare targets for this item
                item_targets = {
                    'rgb': features,  # Add RGB input for edge-aware smoothness loss
                    'depth': targets['depth'][j] if 'depth' in targets and j < len(targets['depth']) else None,
                    'segmentation': targets['segmentation'][j] if 'segmentation' in targets and j < len(targets['segmentation']) else None
                }
                
                # Compute loss
                loss, loss_components = compute_losses(outputs, item_targets, args)
                
                # Compute metrics if targets are available (now using functions from utils)
                if args.predict_depth and 'depth' in outputs and item_targets['depth'] is not None:
                    batch_depth_metrics = compute_depth_metrics(outputs['depth'], item_targets['depth'])
                    for k, v in batch_depth_metrics.items():
                        depth_metrics[k] += v
                    num_depth_samples += 1
                
                if args.predict_segmentation and 'segmentation' in outputs and item_targets['segmentation'] is not None:
                    batch_seg_metrics = compute_segmentation_metrics(outputs['segmentation'], item_targets['segmentation'])
                    for k, v in batch_seg_metrics.items():
                        seg_metrics[k] += v
                    num_seg_samples += 1
                
                batch_loss += loss.item()
                batch_count += 1
                
                # Accumulate loss components
                for component, value in loss_components.items():
                    batch_components[component] += value
            
            # Skip iteration if no valid inputs in batch
            if batch_count == 0:
                continue
            
            # Average loss for this batch
            val_loss += batch_loss / batch_count
            num_batches += 1
            
            # Average loss components
            for component in val_components.keys():
                if component in batch_components and batch_components[component] > 0:
                    val_components[component] += batch_components[component] / batch_count
    
    # Return average validation loss
    if num_batches == 0:
        return 0.0, {}, {}, {}
    
    # Average metrics
    avg_depth_metrics = {}
    if num_depth_samples > 0:
        avg_depth_metrics = {k: v / num_depth_samples for k, v in depth_metrics.items()}
    
    avg_seg_metrics = {}
    if num_seg_samples > 0:
        avg_seg_metrics = {k: v / num_seg_samples for k, v in seg_metrics.items()}
    
    avg_components = {k: v/num_batches for k, v in val_components.items() if v > 0}
    return val_loss / num_batches, avg_components, avg_depth_metrics, avg_seg_metrics
