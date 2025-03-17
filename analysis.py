import torch
import numpy as np
from pathlib import Path

from utils import prepare_batch_for_model

def analyze_depth_predictions(model, test_loader, device, args, checkpoint_dir=None):
    """
    Analyze depth prediction distribution from the model
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run the model on
        args: Command line arguments
        checkpoint_dir: Directory to save output files
        
    Returns:
        dict: Dictionary containing analysis results
    """
    print("\nAnalyzing depth prediction distribution...")
    model.eval()
    depth_predictions = []
    depth_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Skip empty batches
            if not batch or 'rgb' not in batch:
                continue
            
            # Prepare data for model
            rgb_features, edge_indices, rel_poses, targets = prepare_batch_for_model(batch, device, args.distance_threshold)
            
            # Process each item in the batch
            for j, (features, edge_index, rel_pose) in enumerate(zip(rgb_features, edge_indices, rel_poses)):
                # Skip if no edges
                if edge_index.size(1) == 0:
                    continue
                
                # Get predictions
                outputs = model(features, edge_index, rel_pose)
                
                # Collect depth values if available
                if args.predict_depth and 'depth' in outputs and 'depth' in targets and j < len(targets['depth']):
                    pred_depth = outputs['depth'].cpu().numpy().flatten()
                    true_depth = targets['depth'][j].cpu().numpy().flatten()
                    
                    # Filter out invalid values
                    valid_indices = ~np.isnan(true_depth) & (true_depth > 0)
                    if valid_indices.any():
                        depth_predictions.append(pred_depth[valid_indices])
                        depth_targets.append(true_depth[valid_indices])
    
    results = {}
    
    if not depth_predictions:
        print("\nNo valid depth predictions collected for analysis.")
        return results
        
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(depth_predictions)
    all_targets = np.concatenate(depth_targets)
    
    # Calculate statistics
    results = {
        'count': len(all_predictions),
        'min': np.min(all_predictions),
        'max': np.max(all_predictions),
        'mean': np.mean(all_predictions),
        'median': np.median(all_predictions),
        'std': np.std(all_predictions),
    }
    
    # Count zeros and near-zeros
    zero_count = np.sum(np.abs(all_predictions) < 1e-6)
    near_zero_count = np.sum((np.abs(all_predictions) >= 1e-6) & (np.abs(all_predictions) < 0.01))
    results['zero_count'] = zero_count
    results['zero_percentage'] = zero_count / len(all_predictions) * 100
    results['near_zero_count'] = near_zero_count
    results['near_zero_percentage'] = near_zero_count / len(all_predictions) * 100
    
    # Print statistics
    print("\nDepth Prediction Statistics:")
    print(f"  Total depth values: {results['count']:,}")
    print(f"  Min predicted depth: {results['min']:.4f}")
    print(f"  Max predicted depth: {results['max']:.4f}")
    print(f"  Mean predicted depth: {results['mean']:.4f}")
    print(f"  Median predicted depth: {results['median']:.4f}")
    print(f"  Std dev of predicted depth: {results['std']:.4f}")
    print(f"  Zero values (< 1e-6): {zero_count:,} ({results['zero_percentage']:.2f}%)")
    print(f"  Near-zero (< 0.01): {near_zero_count:,} ({results['near_zero_percentage']:.2f}%)")
    
    # Create histogram if checkpoint_dir is provided and matplotlib is available
    if checkpoint_dir is not None:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            # Prediction histogram
            plt.subplot(2, 1, 1)
            plt.hist(all_predictions, bins=50, alpha=0.7, label='Predictions')
            plt.title('Depth Prediction Distribution')
            plt.xlabel('Depth Value')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Target histogram for comparison
            plt.subplot(2, 1, 2)
            plt.hist(all_targets, bins=50, alpha=0.7, color='green', label='Ground Truth')
            plt.title('Ground Truth Depth Distribution')
            plt.xlabel('Depth Value')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # Save the figure
            hist_path = checkpoint_dir / f"depth_histogram_{args.model_type}.png"
            plt.savefig(hist_path)
            print(f"\nDepth histogram saved to {hist_path}")
            
            plt.close()
        except ImportError:
            print("\nMatplotlib not available. Skipping histogram visualization.")
    
    # Store the actual prediction and target arrays in the results
    results['predictions'] = all_predictions
    results['targets'] = all_targets
    
    return results
