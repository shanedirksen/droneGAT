import os
import argparse
import time
import numpy as np
import csv
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data_loader import create_drone_dataloader
from model import DroneGAT
# Import CrossAttention directly
from model_old import CrossAttention
from loss_functions import compute_losses
from trainers import train_epoch, validate  # Import the training functions
from collate import collate_fn  # Import the collate function
from analysis import analyze_depth_predictions  # Import the analysis function
from data_saving import save_checkpoint, save_test_results, save_experiment_results  # Import saving functions
# Import the depth processing function from depth_visualization
from depth_visualization import process_single_depth_map
from utils import (
    build_graph_from_positions, 
    generate_spatial_encoding, 
    calculate_relative_poses,
    compute_depth_metrics,
    compute_segmentation_metrics,
    prepare_batch_for_model
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Drone GAT model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='training/four_drones',
                        help='Path to the dataset directory')
    
    parser.add_argument('--training_corrupt', action='store_true', default=True,
                        help='Randomly corrupt some drones during training')
    
    parser.add_argument('--max_corrupt_ratio', type=float, default=0.5,
                        help='Maximum ratio of drones to corrupt during training (0.0-1.0)')
    
    parser.add_argument('--test_corruption', type=str, default='both',
                        choices=['motion_blur', 'shot_noise', 'both', None],
                        help='Type of corruption to apply during testing (both=randomly select between types)')
    
    parser.add_argument('--test_corruption_mode', type=str, default='partial',
                        choices=['none', 'partial', 'full'],
                        help='Corruption mode for testing (none=0%, partial=33%, full=100%)')
    
    # Removed corruption_intensity parameter as it's now always randomly selected
    
    # Model parameters
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels (3 for RGB)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='Output dimension size')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in GAT')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--distance_threshold', type=float, default=50.0,
                        help='Distance threshold for creating graph edges')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained MobileNetV2 backbone')
    parser.add_argument('--predict_depth', action='store_true', default=True,
                        help='Add depth prediction decoder')
    parser.add_argument('--predict_segmentation', action='store_true', default=True,
                        help='Add segmentation prediction decoder')
    parser.add_argument('--model_type', type=str, default='gat', choices=['gat', 'cross_attention'],
                        help='Model type to use (gat or cross_attention)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.75,
                        help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging training status')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for saving checkpoints')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output during training')
    
    # GPU parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (0 for integrated, 1 for RTX 3090)')
    
    return parser.parse_args()

# Add a function to process depth maps in PyTorch tensor format
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

# Override the prepare_batch_for_model function to process depth maps
def prepare_batch_for_model(batch, device, distance_threshold=50.0):
    """
    Prepare batch data for model input, processing depth maps along the way
    
    Args:
        batch: Batch data dictionary
        device: Device to put tensors on
        distance_threshold: Threshold for building graph
        
    Returns:
        rgb_features, edge_indices, rel_poses, targets
    """
    rgb_features = []
    edge_indices = []
    rel_poses = []
    targets = {}
    
    if not batch:
        return rgb_features, edge_indices, rel_poses, targets
    
    for key in ['depth', 'segmentation']:
        if key in batch:
            targets[key] = []
    
    # Process each item in the batch
    for i, (rgb, positions, orientations) in enumerate(zip(batch['rgb'], batch['positions'], batch['orientations'])):
        # Move data to device
        rgb = rgb.to(device)
        positions = positions.to(device)
        orientations = orientations.to(device)
        
        # Generate edge index (graph connectivity)
        edge_index = build_graph_from_positions(positions, distance_threshold)
        
        # Skip if no edges
        if edge_index.size(1) == 0:
            continue
            
        # Calculate relative poses
        rel_pose = calculate_relative_poses(positions, orientations, edge_index)
        
        rgb_features.append(rgb)
        edge_indices.append(edge_index)
        rel_poses.append(rel_pose)
        
        # Add depth maps if available - process them before adding to targets
        if 'depth' in batch and i < len(batch['depth']):
            depth_maps = batch['depth'][i].to(device)
            # Process depth maps with the same function used in visualization
            processed_depth = process_depth_maps(depth_maps)
            targets['depth'].append(processed_depth)
        
        # Add segmentation masks if available
        if 'segmentation' in batch and i < len(batch['segmentation']):
            seg_masks = batch['segmentation'][i].to(device)
            targets['segmentation'].append(seg_masks)
    
    return rgb_features, edge_indices, rel_poses, targets

# Update the compute_losses function to handle the processed depth maps
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

def visualize_test_samples(model, test_loader, device, args, save_dir, num_samples=5):
    """
    Visualize test samples with their ground truth and predicted depth maps and segmentation maps
    
    Args:
        model: The trained model
        test_loader: DataLoader with test samples
        device: Device to run the model on
        args: Command line arguments
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    print(f"\nGenerating visualizations for {num_samples} test samples...")
    model.eval()
    
    # Make sure the save directory exists
    vis_dir = Path(save_dir) / "test_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the visualizations
    sample_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Skip empty batches
            if not batch or 'rgb' not in batch:
                continue
                
            # Get inputs and outputs for this batch - using the updated prepare_batch_for_model function
            # which now processes depth maps correctly
            rgb_features, edge_indices, rel_poses, targets = prepare_batch_for_model(
                batch, device, args.distance_threshold
            )
            
            # Process each item in the batch
            for i, (features, edge_index, rel_pose) in enumerate(zip(rgb_features, edge_indices, rel_poses)):
                # Skip if no edges or if we've reached our sample limit
                if edge_index.size(1) == 0 or sample_count >= num_samples:
                    continue
                
                # Get predictions from model
                outputs = model(features, edge_index, rel_pose)
                
                # Get ground truth depth and segmentation if available
                depth_gt = None
                seg_gt = None
                if 'depth' in targets and i < len(targets['depth']):
                    depth_gt = targets['depth'][i]
                if 'segmentation' in targets and i < len(targets['segmentation']):
                    seg_gt = targets['segmentation'][i]
                
                # Process a few drones from this sample
                for drone_idx in range(min(features.size(0), 2)):  # Process max 2 drones per sample
                    try:
                        import matplotlib.pyplot as plt
                        
                        # Create a figure with 2x3 subplots (increased from 1x3)
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                        
                        # Plot the input RGB image (denormalize first)
                        rgb_img = features[drone_idx].cpu().permute(1, 2, 0).numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        rgb_img = rgb_img * std + mean
                        rgb_img = np.clip(rgb_img, 0, 1)
                        
                        axes[0, 0].imshow(rgb_img)
                        axes[0, 0].set_title(f"Input RGB (Drone {drone_idx})")
                        axes[0, 0].axis('off')
                        
                        # Plot depth maps
                        if 'depth' in outputs and depth_gt is not None:
                            # Plot ground truth depth - NOTE: already processed by prepare_batch_for_model
                            depth_truth = depth_gt[drone_idx, 0].cpu().numpy()
                            
                            # Plot raw depth values
                            im1 = axes[0, 1].imshow(depth_truth, cmap='viridis')
                            axes[0, 1].set_title("Ground Truth Depth")
                            axes[0, 1].axis('off')
                            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                            
                            # Plot predicted depth - process it as well
                            depth_pred_raw = outputs['depth'][drone_idx, 0].cpu().numpy()
                            # Process the predicted depth the same way
                            depth_pred = process_single_depth_map(depth_pred_raw)
                            im2 = axes[0, 2].imshow(depth_pred, cmap='viridis')
                            axes[0, 2].set_title("Predicted Depth")
                            axes[0, 2].axis('off')
                            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                            
                            # Calculate depth metrics
                            mse_depth = ((depth_truth - depth_pred) ** 2).mean()
                            mae_depth = np.abs(depth_truth - depth_pred).mean()
                        
                        # Plot segmentation maps
                        if 'segmentation' in outputs and seg_gt is not None:
                            # Create a colormap for segmentation
                            cmap = plt.cm.viridis.resampled(3)  # 3 classes
                            
                            # Plot ground truth segmentation
                            if seg_gt[drone_idx].dim() == 3 and seg_gt[drone_idx].size(0) > 1:  # If one-hot encoded
                                seg_truth = torch.argmax(seg_gt[drone_idx], dim=0).cpu().numpy()
                            else:
                                seg_truth = seg_gt[drone_idx].squeeze().cpu().numpy()
                            
                            im3 = axes[1, 1].imshow(seg_truth, cmap=cmap, vmin=0, vmax=2)
                            axes[1, 1].set_title("Ground Truth Segmentation")
                            axes[1, 1].axis('off')
                            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
                            
                            # Plot predicted segmentation (convert one-hot to class indices)
                            seg_pred = torch.argmax(outputs['segmentation'][drone_idx], dim=0).cpu().numpy()
                            im4 = axes[1, 2].imshow(seg_pred, cmap=cmap, vmin=0, vmax=2)
                            axes[1, 2].set_title("Predicted Segmentation")
                            axes[1, 2].axis('off')
                            plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
                            
                            # Calculate segmentation accuracy
                            seg_accuracy = (seg_pred == seg_truth).mean() * 100
                            
                            # Add model information in the bottom-left area
                            axes[1, 0].axis('off')
                            axes[1, 0].text(0.5, 0.5, 
                                        f"Model: {args.model_type.upper()}\n"
                                        f"Distance Threshold: {args.distance_threshold}\n"
                                        f"Num Drones: {features.size(0)}", 
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        bbox=dict(facecolor='white', alpha=0.8))
                        
                        # Add title with metrics
                        metrics_text = ""
                        if 'depth' in outputs and depth_gt is not None:
                            metrics_text += f"Depth - MSE: {mse_depth:.4f}, MAE: {mae_depth:.4f}"
                        if 'segmentation' in outputs and seg_gt is not None:
                            if metrics_text:
                                metrics_text += " | "
                            metrics_text += f"Segmentation - Accuracy: {seg_accuracy:.2f}%"
                        
                        plt.suptitle(f"Sample {sample_count}, Drone {drone_idx}\n{metrics_text}")
                        
                        # Save the figure
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.9)  # Adjust for suptitle
                        fig_path = vis_dir / f"sample_{sample_count}_drone_{drone_idx}.png"
                        plt.savefig(fig_path)
                        plt.close(fig)
                        
                        print(f"Saved visualization to {fig_path}")
                        
                    except ImportError:
                        print("Matplotlib not available. Skipping visualization.")
                        return
                    except Exception as e:
                        print(f"Error during visualization: {str(e)}")
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
    
    print(f"Generated {sample_count} test sample visualizations in {vis_dir}")

def main():
    args = parse_args()
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for available GPUs and print details
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Use the specified GPU or default to the first one
        gpu_id = args.gpu_id if args.gpu_id < num_gpus else 0
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU")
    
    # Image transform applied to all input images
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # MobileNetV2 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    
    # Convert test_corruption_mode to percentage
    test_corruption_percentage = 0
    if args.test_corruption_mode == 'partial':
        test_corruption_percentage = 33
    elif args.test_corruption_mode == 'full':
        test_corruption_percentage = 100
    
    # Create dataset with auto-detection of number of drones
    dataset = create_drone_dataloader(
        root_dir=args.data_dir,
        batch_size=1,
        transform=preprocess,
        corruption_type=None,  # No global corruption during training
        random_corruption=args.training_corrupt,
        max_corrupt_ratio=args.max_corrupt_ratio,
        shuffle=False,
        num_workers=0,
        mode='train',
        test_corruption_percentage=test_corruption_percentage
    ).dataset
    
    # Get the auto-detected number of drones for model creation and reporting
    num_drones = dataset.num_drones
    print(f"Auto-detected {num_drones} drones in the dataset")
    
    # Calculate split sizes based on dataset length and specified ratios
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_split)
    val_size = int(dataset_size * args.val_split)
    test_size = dataset_size - train_size - val_size
    
    # Ensure we have at least some data in each split
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(f"Invalid split ratios. Resulting splits: train={train_size}, val={val_size}, test={test_size}")
        
    # Split dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # For validation, we want consistent corruption behavior
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create a separate dataloader for testing with specified corruption settings
    test_dataset_with_corruption = create_drone_dataloader(
        root_dir=args.data_dir,
        batch_size=1,
        transform=preprocess,
        corruption_type=args.test_corruption if args.test_corruption_mode != 'none' else None,
        random_corruption=False,  # No randomness in which drones are corrupted during testing
        max_corrupt_ratio=0.0,    # Not used for testing
        shuffle=False,
        num_workers=0,
        mode='test',
        test_corruption_percentage=test_corruption_percentage  # Apply the correct percentage
    ).dataset
    
    # Use the same indices as the original test split
    test_indices = list(range(len(dataset)))[train_size + val_size:]
    test_corruption_dataset = torch.utils.data.Subset(test_dataset_with_corruption, test_indices)
    
    test_loader = DataLoader(
        test_corruption_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Dataset split: Total={dataset_size}, Training={train_size}, Validation={val_size}, Test={test_size}")
    print(f"Training with random corruption: {args.training_corrupt} (max ratio: {args.max_corrupt_ratio*100:.0f}%)")
    print(f"Testing with corruption: {args.test_corruption} "
          f"(mode: {args.test_corruption_mode}, using randomly selected intensities)")
    
    # Initialize the model with all components based on model_type
    if args.model_type == 'gat':
        print("Using Graph Attention Network (GAT) model")
        model = DroneGAT(
            input_channels=args.input_channels,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pretrained=args.pretrained,
            predict_depth=args.predict_depth,
            predict_segmentation=args.predict_segmentation
        ).to(device)
    else:  # cross_attention
        print("Using Cross Attention model")
        model = CrossAttention(  # Use CrossAttention directly
            input_channels=args.input_channels,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pretrained=args.pretrained,
            predict_depth=args.predict_depth,
            predict_segmentation=args.predict_segmentation
        ).to(device)
    
    print(f"Model created with the following tasks:")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Depth prediction: {args.predict_depth}")
    print(f"  - Segmentation: {args.predict_segmentation}")
    print(f"  - Pretrained backbone: {args.pretrained}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create lists to track metrics over time
    train_losses = []
    val_losses = []
    
    depth_metrics_history = {
        'abs_rel': [],
        'sq_rel': [],
        'rmse': []
    }
    
    seg_metrics_history = {
        'miou': [],
        'pixel_acc': []
    }
    
    # Train the model
    print("\nStarting training...")
    
    # Display headers with category labels
    print("\nTraining Progress:")
    print("-" * 100)
    print(f"{'':^6} | {'':^10} | {'':^10} | {'DEPTH ESTIMATION METRICS':^32} | {'SEGMENTATION METRICS':^21} | {'':^8}")
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Abs Rel↓':^10} {'Sq Rel↓':^10} {'RMSE↓':^10} | {'mIoU↑':^10} {'Pixel Acc↑':^10} | {'Time (s)':^8}")
    print("-" * 100)
    
    epoch_results = []
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train for one epoch - using the imported function
        train_loss, train_components = train_epoch(model, train_loader, optimizer, device, args)
        train_losses.append(train_loss)
        
        # Validate - using the imported function
        val_loss, val_components, depth_metrics, seg_metrics = validate(model, val_loader, device, args)
        val_losses.append(val_loss)
        
        # Update metrics history
        for k, v in depth_metrics.items():
            if k in depth_metrics_history:
                depth_metrics_history[k].append(v)
                
        for k, v in seg_metrics.items():
            if k in seg_metrics_history:
                seg_metrics_history[k].append(v)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Format metrics for display
        abs_rel = depth_metrics.get('abs_rel', float('nan'))
        sq_rel = depth_metrics.get('sq_rel', float('nan'))
        rmse = depth_metrics.get('rmse', float('nan'))
        miou = seg_metrics.get('miou', float('nan'))
        pixel_acc = seg_metrics.get('pixel_acc', float('nan'))
        
        # Store result for this epoch
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'miou': miou,
            'pixel_acc': pixel_acc,
            'time': epoch_time
        }
        epoch_results.append(epoch_result)
        
        # Print epoch results in table format
        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{abs_rel:>10.4f} {sq_rel:>10.4f} {rmse:>10.4f} | "
              f"{miou:>10.4f} {pixel_acc:>10.4f} | "
              f"{epoch_time:>8.2f}")
        
        # Save checkpoint using the imported function
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                checkpoint_dir, 
                model, 
                optimizer, 
                epoch + 1, 
                train_loss,
                val_loss, 
                depth_metrics, 
                seg_metrics,
                args.model_type,
                interval=True,
                verbose=args.verbose
            )
    
    print("-" * 100)
    print("Training completed!")
    
    # Print final metrics
    print("\nFinal Metrics Summary:")
    print("Depth Estimation Metrics (lower is better):")
    print(f"  Absolute Relative Difference (Abs Rel): {depth_metrics.get('abs_rel', float('nan')):.4f}")
    print(f"  Squared Relative Difference  (Sq Rel): {depth_metrics.get('sq_rel', float('nan')):.4f}")
    print(f"  Root Mean Square Error       (RMSE)  : {depth_metrics.get('rmse', float('nan')):.4f}")
    
    print("\nSemantic Segmentation Metrics (higher is better):")
    print(f"  Mean Intersection over Union (mIoU)     : {seg_metrics.get('miou', float('nan')):.4f}")
    print(f"  Pixel Accuracy                         : {seg_metrics.get('pixel_acc', float('nan')):.4f}")
    
    # Save final model using the imported function
    save_checkpoint(checkpoint_dir, model, optimizer, args.num_epochs, train_loss, val_loss, 
                   depth_metrics, seg_metrics, args.model_type, interval=False)
    
    # After training, evaluate on the test set with specified corruption
    print("\nEvaluating on test set with specified corruption settings...")
    print(f"Test corruption: {args.test_corruption}, Mode: {args.test_corruption_mode}")
    test_loss, test_components, test_depth_metrics, test_seg_metrics = validate(model, test_loader, device, args)
    
    print("\nTest Set Metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print("Depth Estimation Metrics (lower is better):")
    print(f"  Absolute Relative Difference (Abs Rel): {test_depth_metrics.get('abs_rel', float('nan')):.4f}")
    print(f"  Squared Relative Difference  (Sq Rel): {test_depth_metrics.get('sq_rel', float('nan')):.4f}")
    print(f"  Root Mean Square Error       (RMSE)  : {test_depth_metrics.get('rmse', float('nan')):.4f}")
    
    print("\nSemantic Segmentation Metrics (higher is better):")
    print(f"  Mean Intersection over Union (mIoU)     : {test_seg_metrics.get('miou', float('nan')):.4f}")
    print(f"  Pixel Accuracy                         : {test_seg_metrics.get('pixel_acc', float('nan')):.4f}")
    
    # Save test results using the imported function
    save_test_results(checkpoint_dir, test_loss, test_depth_metrics, test_seg_metrics, args.model_type)
    
    # Save experiment results to CSV using the imported function
    save_experiment_results(args, test_loss, test_depth_metrics, test_seg_metrics, 
                          num_drones, test_corruption_percentage)
    
    # Use the imported analysis function to analyze depth predictions
    depth_analysis_results = analyze_depth_predictions(model, test_loader, device, args, checkpoint_dir)
    
    # Add this line right before or after the depth analysis
    visualize_test_samples(model, test_loader, device, args, checkpoint_dir)

if __name__ == '__main__':
    main()