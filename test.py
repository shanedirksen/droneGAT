import os
import argparse
import time
import numpy as np
import csv
from pathlib import Path
import sys
import matplotlib.pyplot as plt  # Add this import

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import create_drone_dataloader, read_pfm
from model import DroneGAT
# Import CrossAttention directly
from model_old import CrossAttention
from utils import (
    compute_depth_metrics,
    compute_segmentation_metrics,
    prepare_batch_for_model
)

# Fix the write_pfm function to handle different input shapes
def write_pfm(file, image, scale=1):
    """Write depth data to PFM file format."""
    # First ensure image is numpy array
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    
    # Check shape and reshape appropriately
    if image.ndim == 1:
        # If we have a flattened array, reshape to square (assuming it's a square)
        size = int(np.sqrt(image.size))
        image = image.reshape(size, size)
    elif image.ndim == 3 and image.shape[0] == 1:
        # If we have shape [1, H, W], convert to [H, W]
        image = image.squeeze(0)
    
    with open(file, 'wb') as f:
        color = None
        
        if image.ndim == 3 and image.shape[2] == 3:
            color = True
        else:
            color = False
            # Reshape only if not already in proper shape
            if image.ndim == 2:
                # Already 2D, add channel dimension
                image = image.reshape(image.shape[0], image.shape[1], 1)
        
        f.write(b'PF\n' if color else b'Pf\n')
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        
        # Negative scale means little-endian
        endian = image.dtype.byteorder
        scale = -scale if (endian == '<' or endian == '=' and sys.byteorder == 'little') else scale
        f.write(f"{scale}\n".encode())
        
        # Convert to float32 if needed
        image = image.astype(np.float32)
        
        # PFM stores images upside-down
        image = np.flipud(image)
        
        # Write the data
        image.tofile(f)

# Add function to save segmentation as color image
def save_segmentation(file, segmentation):
    """Save segmentation prediction as a color-coded image."""
    # Convert from one-hot or class indices to a color map
    if len(segmentation.shape) > 2 and segmentation.shape[0] > 1:
        # One-hot encoded - get class indices
        seg_indices = np.argmax(segmentation, axis=0)
    else:
        # Already class indices
        seg_indices = segmentation.squeeze()
    
    # Create a simple color map (can be customized based on your classes)
    # Map each class index to a unique color
    color_map = {}
    unique_indices = np.unique(seg_indices)
    colors = np.linspace(0, 255, len(unique_indices), dtype=np.uint8)
    
    for i, idx in enumerate(unique_indices):
        color_map[idx] = colors[i]
    
    # Create RGB image
    height, width = seg_indices.shape
    seg_colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    for idx in unique_indices:
        mask = (seg_indices == idx)
        seg_colored[mask] = [color_map[idx], color_map[idx], color_map[idx]]
    
    # Save as image
    import cv2
    cv2.imwrite(str(file), seg_colored)

# Add function to generate depth histograms
def generate_depth_histogram(depth_values, num_bins, max_depth, output_path):
    """Generate a histogram of depth values and save it to a file"""
    # Convert to numpy array if not already
    if torch.is_tensor(depth_values):
        depth_values = depth_values.detach().cpu().numpy()
    depth_values = np.array(depth_values).flatten()
    
    # Filter out invalid values (negative, zero, or very large values)
    depth_values = depth_values[(depth_values > 0) & (depth_values < max_depth)]
    
    if len(depth_values) == 0:
        print("Warning: No valid depth values found for histogram")
        return
    
    # Define bins
    bins = np.linspace(0, max_depth, num_bins + 1)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(depth_values, bins=bins)
    
    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw histogram data
    hist_data_path = output_path / 'depth_histogram.csv'
    with open(hist_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_min', 'bin_max', 'count'])
        for i in range(len(hist)):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            writer.writerow([bin_min, bin_max, hist[i]])
    
    print(f"Saved depth histogram data to {hist_data_path}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=(max_depth/num_bins), alpha=0.7, 
            edgecolor='black', align='edge')
    plt.xlabel('Depth (m)')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Depths')
    plt.grid(axis='y', alpha=0.75)
    
    # Add summary statistics
    plt.text(0.7, 0.9, f"Mean: {np.mean(depth_values):.2f}m\n" +
             f"Median: {np.median(depth_values):.2f}m", 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    hist_plot_path = output_path / 'depth_histogram.png'
    plt.savefig(hist_plot_path)
    plt.close()
    print(f"Saved depth histogram plot to {hist_plot_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test Drone GAT model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='training/four_drones',
                        help='Path to the dataset directory')
    
    parser.add_argument('--test_corruption', type=str, default='both',
                        choices=['motion_blur', 'shot_noise', 'both', None],
                        help='Type of corruption to apply during testing (both=randomly select between types)')
    
    parser.add_argument('--test_corruption_mode', type=str, default='partial',
                        choices=['none', 'partial', 'full'],
                        help='Corruption mode for testing (none=0%, partial=33%, full=100%)')
    
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
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Testing batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Model loading parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    
    # GPU parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (0 for integrated, 1 for RTX 3090)')
    
    # Add new argument for saving predictions
    parser.add_argument('--save_predictions', action='store_true', default=False,
                        help='Save model predictions as files')
    parser.add_argument('--prediction_dir', type=str, default='predictions',
                        help='Directory to save prediction outputs')
    
    # Add depth histogram arguments
    parser.add_argument('--generate_depth_histogram', action='store_true',
                        help='Generate histogram of depth predictions')
    parser.add_argument('--histogram_bins', type=int, default=20, 
                        help='Number of histogram bins for depth distribution')
    parser.add_argument('--histogram_max_depth', type=float, default=100.0,
                        help='Maximum depth value for histogram (meters)')

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

def validate(model, dataloader, device, args, save_predictions=False, prediction_dir=None):
    """
    Validate the model and optionally save predictions
    """
    model.eval()
    
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
        for batch_idx, batch in enumerate(dataloader):
            # Skip empty batches
            if not batch or 'rgb' not in batch:
                continue
            
            # Get timestamp from batch
            timestamp = batch['timestamps'][0]
            
            # Prepare data for model
            rgb_features, edge_indices, rel_poses, targets = prepare_batch_for_model(batch, device, args.distance_threshold)
            
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
                    'rgb': features,
                    'depth': targets['depth'][j] if 'depth' in targets and j < len(targets['depth']) else None,
                    'segmentation': targets['segmentation'][j] if 'segmentation' in targets and j < len(targets['segmentation']) else None
                }
                
                # Save predictions if requested
                if save_predictions and prediction_dir is not None:
                    # Save depth predictions
                    if 'depth' in outputs:
                        depth_pred = outputs['depth'].detach().cpu()
                        
                        # Create a directory structure similar to the input data
                        timestamp_dir = prediction_dir / f"depth/timestamp_{timestamp:.6f}"
                        timestamp_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Print shape info for debugging
                        if batch_idx == 0 and j == 0:
                            print(f"Debug - Depth prediction shape: {depth_pred.shape}")
                        
                        # Save one depth map for each drone
                        for i in range(depth_pred.shape[0]):  # For each drone in this prediction
                            # Get the depth prediction for this drone
                            drone_depth = depth_pred[i]
                            
                            # Save as PFM file
                            depth_file = timestamp_dir / f"drone_{i}.pfm"
                            write_pfm(depth_file, drone_depth)
                            
                            # Only print for first few files to avoid flooding console
                            if batch_idx < 2:
                                print(f"Saved depth prediction to {depth_file}")
                    
                    # Save segmentation predictions
                    if 'segmentation' in outputs:
                        seg_pred = outputs['segmentation'].detach().cpu().numpy()
                        
                        # Create directory
                        timestamp_dir = prediction_dir / f"segmentation/timestamp_{timestamp:.6f}"
                        timestamp_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save each segmentation map
                        for i in range(seg_pred.shape[0]):  # For each drone in this prediction
                            # Generate unique identifier
                            drone_id = f"batch{batch_idx}_drone{i}"
                            
                            # Get the segmentation prediction
                            drone_seg = seg_pred[i]
                            
                            # Save as image file
                            seg_file = timestamp_dir / f"drone_{i}.png"
                            save_segmentation(seg_file, drone_seg)
                            
                            print(f"Saved segmentation prediction to {seg_file}")
                
                # Compute metrics if targets are available
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
                
                batch_count += 1
            
            # Skip iteration if no valid inputs in batch
            if batch_count == 0:
                continue
            
            num_batches += 1
    
    # Average metrics
    avg_depth_metrics = {}
    if num_depth_samples > 0:
        avg_depth_metrics = {k: v / num_depth_samples for k, v in depth_metrics.items()}
    
    avg_seg_metrics = {}
    if num_seg_samples > 0:
        avg_seg_metrics = {k: v / num_seg_samples for k, v in seg_metrics.items()}
    
    return avg_depth_metrics, avg_seg_metrics

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create prediction directory if saving predictions
    prediction_dir = None
    if args.save_predictions:
        prediction_dir = Path(args.prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model predictions will be saved to {prediction_dir}")
    
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
    
    # Create dataset with test corruption settings
    test_dataset = create_drone_dataloader(
        root_dir=args.data_dir,
        batch_size=1,
        transform=preprocess,
        corruption_type=args.test_corruption if args.test_corruption_mode != 'none' else None,
        random_corruption=False,  # No randomness in which drones are corrupted during testing
        max_corrupt_ratio=0.0,    # Not used for testing
        shuffle=False,
        num_workers=0,
        mode='test',
        test_corruption_percentage=test_corruption_percentage
    ).dataset
    
    # Print dataset information
    num_drones = test_dataset.num_drones
    print(f"Test dataset: {len(test_dataset)} samples with {num_drones} drones")
    print(f"Testing with corruption: {args.test_corruption} "
          f"(mode: {args.test_corruption_mode}, {test_corruption_percentage}%)")
    
    # Create the test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize the model based on model_type
    if args.model_type == 'gat':
        print("Loading Graph Attention Network (GAT) model")
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
        print("Loading Cross Attention model")
        model = CrossAttention(
            input_channels=args.input_channels,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pretrained=args.pretrained,
            predict_depth=args.predict_depth,
            predict_segmentation=args.predict_segmentation
        ).to(device)
    
    # Load model weights
    print(f"Loading model weights from {args.model_path}")
    model_state = torch.load(args.model_path, map_location=device)
    
    # Handle different saved formats (full checkpoint vs just state_dict)
    if isinstance(model_state, dict) and 'model_state_dict' in model_state:
        model.load_state_dict(model_state['model_state_dict'])
        print(f"Loaded checkpoint from epoch {model_state.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(model_state)
        print("Loaded model state dictionary")
    
    print(f"Model configuration:")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Depth prediction: {args.predict_depth}")
    print(f"  - Segmentation: {args.predict_segmentation}")
    print(f"  - Pretrained backbone: {args.pretrained}")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    start_time = time.time()
    test_depth_metrics, test_seg_metrics = validate(
        model, 
        test_loader, 
        device, 
        args,
        save_predictions=args.save_predictions,
        prediction_dir=prediction_dir
    )
    test_time = time.time() - start_time
    
    # Print results
    print(f"\nTest completed in {test_time:.2f} seconds")
    print("\nTest Set Metrics:")
    
    print("Depth Estimation Metrics (lower is better):")
    print(f"  Absolute Relative Difference (Abs Rel): {test_depth_metrics.get('abs_rel', float('nan')):.4f}")
    print(f"  Squared Relative Difference  (Sq Rel): {test_depth_metrics.get('sq_rel', float('nan')):.4f}")
    print(f"  Root Mean Square Error       (RMSE)  : {test_depth_metrics.get('rmse', float('nan')):.4f}")
    
    print("\nSemantic Segmentation Metrics (higher is better):")
    print(f"  Mean Intersection over Union (mIoU)     : {test_seg_metrics.get('miou', float('nan')):.4f}")
    print(f"  Pixel Accuracy                         : {test_seg_metrics.get('pixel_acc', float('nan')):.4f}")
    
    if args.save_predictions:
        print(f"\nModel predictions saved to {prediction_dir}")
        print("Prediction files are organized by type (depth/segmentation) and timestamp")
    
    # Create filename with test parameters
    model_name = Path(args.model_path).stem
    result_filename = f"test_{model_name}_{args.test_corruption}_{args.test_corruption_mode}.txt"
    test_results_path = output_dir / result_filename
    
    # Save test metrics to a file
    with open(test_results_path, 'w') as f:
        f.write(f"Test Results for model: {args.model_path}\n")
        f.write(f"Test configuration: corruption={args.test_corruption}, mode={args.test_corruption_mode}, percentage={test_corruption_percentage}%\n")
        f.write(f"Test time: {test_time:.2f} seconds\n\n")
        
        f.write("Depth Estimation Metrics:\n")
        for k, v in test_depth_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nSegmentation Metrics:\n")
        for k, v in test_seg_metrics.items():
            f.write(f"  {v:.4f}\n")
    
    print(f"\nTest results saved to {test_results_path}")
    
    # Also save the results to a CSV file for tracking experiments
    csv_path = output_dir / 'test_results.csv'
    
    # Prepare row data
    row_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': args.model_path,
        'model_type': args.model_type,
        'test_corruption_type': args.test_corruption,
        'test_corruption_mode': args.test_corruption_mode,
        'test_corruption_percentage': test_corruption_percentage,
    }
    
    # Add depth metrics
    for k, v in test_depth_metrics.items():
        row_data[f'depth_{k}'] = v
    
    # Add segmentation metrics
    for k, v in test_seg_metrics.items():
        row_data[f'seg_{k}'] = v
    
    # Write to CSV - create new file with headers if it doesn't exist
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"Test results appended to {csv_path}")

if __name__ == '__main__':
    main()
