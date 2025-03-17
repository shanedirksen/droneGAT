import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data_loader import create_drone_dataloader
from model import DroneGAT
from utils import (
    build_graph_from_positions, 
    calculate_relative_poses,
    visualize_distance_matrix,
    visualize_3d_graph,
    visualize_2d_graph
)
from depth_visualization import process_single_depth_map, visualize_depth_map

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Drone GAT Model and Graph Structure')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--corruption_type', type=str, default=None, 
                        choices=['motion_blur', 'shot_noise', 'both', None],
                        help='Type of corruption to apply')
    parser.add_argument('--test_corruption_percentage', type=int, default=33,
                        choices=[0, 33, 100],
                        help='Percentage of drones to corrupt (0, 33, or 100)')
    
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
    
    # Checkpoint parameters
    parser.add_argument('--model_path', type=str, default='checkpoints/model_final.pt',
                        help='Path to the saved model checkpoint')
    parser.add_argument('--model_type', type=str, default='gat', choices=['gat', 'cross_attention'],
                        help='Type of model architecture (gat or cross_attention)')
                        
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--drone_idx', type=int, default=0,
                        help='Index of the drone to visualize (0-3 for four_drones)')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()

def collate_fn(batch):
    """
    Custom collate function for handling drone data and creating graph structure
    """
    batch_data = {}
    
    # Collect all data from all drones in the batch
    rgb_images = []
    depth_images = []
    seg_images = []
    positions = []
    orientations = []
    timestamps = []
    adjacency_matrices = []
    
    for item in batch:
        drone_features = item['drone_features']
        
        # Extract data from each drone
        batch_rgb = []
        batch_depth = []
        batch_seg = []
        batch_pos = []
        batch_ori = []
        
        for drone in drone_features:
            if drone['rgb'] is not None:
                batch_rgb.append(drone['rgb'])
                batch_pos.append(drone['position'])
                batch_ori.append(drone['orientation'])
                
                # Add depth and segmentation if available
                if 'depth' in drone and drone['depth'] is not None:
                    batch_depth.append(drone['depth'])
                if 'segmentation' in drone and drone['segmentation'] is not None:
                    batch_seg.append(drone['segmentation'])
        
        # Convert to tensors
        if batch_rgb:
            rgb_tensor = torch.stack(batch_rgb)
            pos_tensor = torch.stack(batch_pos)
            ori_tensor = torch.stack(batch_ori)
            
            rgb_images.append(rgb_tensor)
            positions.append(pos_tensor)
            orientations.append(ori_tensor)
            timestamps.append(item['timestamp'])
            adjacency_matrices.append(item['adjacency'])
            
            if batch_depth:
                depth_tensor = torch.stack(batch_depth)
                depth_images.append(depth_tensor)
                
            if batch_seg:
                seg_tensor = torch.stack(batch_seg)
                seg_images.append(seg_tensor)
    
    # Stack all tensors
    if rgb_images:
        batch_data['rgb'] = rgb_images
        batch_data['positions'] = positions
        batch_data['orientations'] = orientations
        batch_data['timestamps'] = timestamps
        batch_data['adjacency'] = adjacency_matrices
        
        if depth_images:
            batch_data['depth'] = depth_images
        if seg_images:
            batch_data['segmentation'] = seg_images
        
    return batch_data

def detect_num_drones(data_dir):
    """
    Automatically detect the number of drones based on subdirectories in data_dir
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Number of drones detected
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist.")
        return 4  # Default to 4 if directory doesn't exist
    
    # Count the number of drone directories
    drone_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('Drone')]
    
    num_drones = len(drone_dirs)
    
    if num_drones == 0:
        print(f"Warning: No drone directories found in {data_dir}, defaulting to 4 drones.")
        return 4
    
    print(f"Detected {num_drones} drones in dataset directory.")
    return num_drones

def visualize_model_predictions(model, dataloader, device, args):
    """
    Visualize model predictions compared to ground truth
    
    Args:
        model: The trained model to use for predictions
        dataloader: The dataloader containing the dataset samples
        device: Device to run the model on
        args: Command line arguments
    """
    model.eval()
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    samples_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Skip empty batches
            if not batch or 'rgb' not in batch:
                continue
                
            # Prepare data for the specified drone
            for i, (rgb, positions, orientations) in enumerate(zip(batch['rgb'], batch['positions'], batch['orientations'])):
                # Move data to device
                rgb = rgb.to(device)
                positions = positions.to(device)
                orientations = orientations.to(device)
                
                # Generate edge index (graph connectivity)
                edge_index = build_graph_from_positions(positions, args.distance_threshold)
                
                # Skip if no edges
                if edge_index.size(1) == 0:
                    continue
                    
                # Calculate relative poses
                rel_pose = calculate_relative_poses(positions, orientations, edge_index)
                
                # Forward pass
                outputs = model(rgb, edge_index, rel_pose)
                
                # Get ground truth if available
                depth_gt = None
                seg_gt = None
                if 'depth' in batch and i < len(batch['depth']):
                    depth_gt = batch['depth'][i].to(device)
                if 'segmentation' in batch and i < len(batch['segmentation']):
                    seg_gt = batch['segmentation'][i].to(device)
                
                # Focus on the specified drone
                drone_idx = min(args.drone_idx, rgb.size(0)-1)
                
                # Get inputs and outputs for the specified drone
                input_rgb = rgb[drone_idx].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                
                # Denormalize RGB image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_rgb = input_rgb * std + mean
                input_rgb = np.clip(input_rgb, 0, 1)
                
                # Create figure for this sample
                fig = plt.figure(figsize=(15, 10))
                
                # Plot input RGB
                plt.subplot(2, 3, 1)
                plt.title('Input RGB')
                plt.imshow(input_rgb)
                plt.axis('off')
                
                # Plot depth prediction if available
                if 'depth' in outputs:
                    # Convert prediction to numpy
                    depth_pred = outputs['depth'][drone_idx, 0].cpu().numpy()
                    
                    # Process and visualize depth prediction using the shared function
                    processed_depth_pred = process_single_depth_map(depth_pred)
                    ax_pred = plt.subplot(2, 3, 2)
                    _, _, im_pred = visualize_depth_map(
                        processed_depth_pred, 
                        ax=ax_pred,
                        title='Depth Prediction'
                    )
                    
                    # Plot ground truth depth if available
                    if depth_gt is not None:
                        depth_truth = depth_gt[drone_idx, 0].cpu().numpy()
                        
                        # Process and visualize ground truth depth
                        processed_depth_truth = process_single_depth_map(depth_truth)
                        ax_gt = plt.subplot(2, 3, 3)
                        _, _, im_gt = visualize_depth_map(
                            processed_depth_truth,
                            ax=ax_gt,
                            title='Ground Truth Depth'
                        )
                        
                        # Calculate and display error metrics
                        mse = ((processed_depth_truth - processed_depth_pred) ** 2).mean()
                        mae = np.abs(processed_depth_truth - processed_depth_pred).mean()
                        plt.figtext(0.5, 0.02, f'MSE: {mse:.4f}, MAE: {mae:.4f}', 
                                   ha='center', fontsize=12, 
                                   bbox=dict(facecolor='white', alpha=0.8))
                
                # Plot segmentation prediction if available
                if 'segmentation' in outputs:
                    # Convert one-hot to class indices
                    seg_pred = torch.argmax(outputs['segmentation'][drone_idx], dim=0).cpu().numpy()
                    
                    # Create a colormap for segmentation
                    cmap = plt.colormaps['viridis'].resampled(3)  # 3 classes
                    
                    plt.subplot(2, 3, 5)
                    plt.title('Segmentation Prediction')
                    plt.imshow(seg_pred, cmap=cmap, vmin=0, vmax=2)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.axis('off')
                    
                    # Plot ground truth segmentation if available
                    if seg_gt is not None:
                        if seg_gt.shape[1] == 3:  # If one-hot encoded
                            seg_truth = torch.argmax(seg_gt[drone_idx], dim=0).cpu().numpy()
                        else:
                            seg_truth = seg_gt[drone_idx, 0].cpu().numpy()
                            
                        plt.subplot(2, 3, 6)
                        plt.title('Ground Truth Segmentation')
                        plt.imshow(seg_truth, cmap=cmap, vmin=0, vmax=2)
                        plt.colorbar(fraction=0.046, pad=0.04)
                        plt.axis('off')
                
                # Add title with timestamp and drone info
                timestamp = batch['timestamps'][i]
                plt.suptitle(f'Drone {drone_idx} Predictions (Timestamp: {timestamp:.2f})', fontsize=16)
                
                # Add model information in the bottom-left area (subplot 2,3,4 area)
                plt.subplot(2, 3, 4)
                plt.axis('off')
                plt.text(0.5, 0.5, 
                         f"Model: {args.model_type.upper()}\n"
                         f"Model Path: {Path(args.model_path).name}\n"
                         f"Threshold: {args.distance_threshold}\n"
                         f"Num Drones: {positions.size(0)}", 
                         horizontalalignment='center',
                         verticalalignment='center',
                         bbox=dict(facecolor='white', alpha=0.8))
                
                # Save figure
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)  # Make room for suptitle
                plt.savefig(save_dir / f'sample_{samples_count}_drone_{drone_idx}.png')
                plt.close(fig)
                
                samples_count += 1
                if samples_count >= args.num_samples:
                    return

def visualize_graph_structure(dataloader, args):
    """
    Visualize the graph structure of drones based on their positions
    
    Args:
        dataloader: The dataloader containing the dataset samples
        args: Command line arguments
    """
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process one batch for graph visualization
    for batch_idx, batch in enumerate(dataloader):
        # Skip empty batches
        if not batch or 'rgb' not in batch:
            continue
        
        # Get the first item with valid data
        for i, (positions, timestamp) in enumerate(zip(batch['positions'], batch['timestamps'])):
            # Create distance matrix visualization
            visualize_distance_matrix(
                positions=positions,
                distance_threshold=args.distance_threshold,
                timestamp=timestamp,
                save_path=save_dir / f'distance_matrix_timestamp_{timestamp:.2f}.png'
            )
            
            # Build the graph
            edge_index = build_graph_from_positions(positions, args.distance_threshold)
            
            # Create 3D graph visualization
            visualize_3d_graph(
                positions=positions,
                edge_index=edge_index,
                distance_threshold=args.distance_threshold,
                timestamp=timestamp,
                save_path=save_dir / f'graph_3d_timestamp_{timestamp:.2f}.png'
            )
            
            # Create 2D graph visualization
            visualize_2d_graph(
                positions=positions,
                edge_index=edge_index,
                distance_threshold=args.distance_threshold,
                timestamp=timestamp,
                save_path=save_dir / f'graph_2d_timestamp_{timestamp:.2f}.png'
            )
            
            # Only process the first item
            return

def main():
    args = parse_args()
    
    # Set device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Use the specified GPU
        gpu_id = args.gpu_id if args.gpu_id < num_gpus else 0
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU")
    
    print(f"Using dataset from: {args.data_dir}")
    
    # Auto-detect number of drones from the dataset directory
    detected_num_drones = detect_num_drones(args.data_dir)
    
    # Image transform applied to all input images
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # MobileNetV2 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    
    # Create dataset
    dataset = create_drone_dataloader(
        root_dir=args.data_dir,
        batch_size=1,  
        transform=preprocess,  
        corruption_type=args.corruption_type,  # Updated parameter name
        shuffle=False,
        num_workers=0,
        mode='test',
        test_corruption_percentage=args.test_corruption_percentage
    ).dataset
    
    # Split into training and validation sets for consistent comparison
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Create data loader
    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,  # Shuffle to get varied samples
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Initialize model based on the model type
    if args.model_type == 'gat':
        print(f"Using Graph Attention Network (GAT) model architecture")
        model = DroneGAT(
            input_channels=args.input_channels,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pretrained=False,  # No need to download weights again, we'll load from checkpoint
            predict_depth=True,
            predict_segmentation=True
        ).to(device)
    else:  # cross_attention
        print(f"Using Cross Attention model architecture")
        from model_old import CrossAttention
        model = CrossAttention(
            input_channels=args.input_channels,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pretrained=False,
            predict_depth=True,
            predict_segmentation=True
        ).to(device)
    
    # Load model weights
    if os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at {args.model_path}")
    
    # First, visualize the graph structure
    print(f"Generating graph visualization...")
    visualize_graph_structure(dataloader, args)
    print(f"Graph visualization saved to {args.save_dir}")
    
    # Then, visualize model predictions for multiple samples
    print(f"Generating model prediction visualizations for {args.num_samples} samples using {args.model_type.upper()} model...")
    visualize_model_predictions(model, dataloader, device, args)
    print(f"Model visualizations saved to {args.save_dir}")

if __name__ == '__main__':
    main()
