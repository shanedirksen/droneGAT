import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import argparse
from pathlib import Path
import glob

def read_pfm(file):
    """Read a PFM file."""
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF': color = True
        elif header == 'Pf': color = False
        else: raise ValueError('Not a PFM file.')
        
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match: width, height = map(int, dim_match.groups())
        else: raise ValueError('Malformed PFM header.')
        
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape)

def write_pfm(file, data):
    """
    Write a PFM file.
    
    Args:
        file: Path to the output file
        data: Numpy array containing the depth data
    """
    with open(file, 'wb') as f:
        # Write the header: PF for color images, Pf for grayscale
        header = 'Pf\n'
        f.write(header.encode('utf-8'))
        
        # Write dimensions: width, height
        height, width = data.shape
        f.write(f'{width} {height}\n'.encode('utf-8'))
        
        # Write scale factor (negative for little endian)
        scale = -1.0  # Use little endian
        f.write(f'{scale}\n'.encode('utf-8'))
        
        # Write data
        # Convert to float32 to match the expected format
        data_float32 = data.astype(np.float32)
        data_float32.tofile(f)

def get_corresponding_depth_file(rgb_path):
    """Find the corresponding depth file for an RGB image."""
    rgb_path = Path(rgb_path)
    # We need to change the third number in the filename from 0 to 1
    # Example: img_Drone0_0_0.6400000000000101.png -> img_Drone0_1_0.6400000000000101.pfm
    
    filename_parts = rgb_path.stem.split('_')
    if len(filename_parts) >= 4:
        # Construct the depth filename
        depth_filename = f"{filename_parts[0]}_{filename_parts[1]}_1_{filename_parts[3]}.pfm"
        depth_path = rgb_path.parent / depth_filename
        return depth_path
    return None

def get_model_depth_file(timestamp, drone_id):
    """Find the model depth estimation file for a given timestamp and drone ID."""
    # Extract the drone number from the drone ID (e.g., "Drone0" -> "0")
    drone_num = drone_id.replace("Drone", "")
    
    # Format the timestamp for the directory name
    timestamp_formatted = float(timestamp)
    timestamp_dir = f"timestamp_{timestamp_formatted:.6f}"
    
    # Construct the path to the model depth file
    base_path = Path("C:/Users/shane/Documents/GitHub/drone_gat/predictions/depth")
    model_depth_path = base_path / timestamp_dir / f"drone_{drone_num}.pfm"
    
    if model_depth_path.exists():
        return model_depth_path
    else:
        return None

def process_drone_folder(base_dir, drone_id=None):
    """
    Process all images in a specific drone folder or in all drone folders.
    
    Args:
        base_dir: Base directory containing drone folders (e.g., "training/four_drones")
        drone_id: Specific drone ID to process (e.g., "Drone0"). If None, process all drones.
    """
    # Create output directory if it doesn't exist
    output_dir = Path("depth_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return
    
    # Get list of drone folders
    if drone_id:
        drone_folders = [base_path / drone_id]
        if not drone_folders[0].exists():
            print(f"Error: Drone folder {drone_id} does not exist in {base_dir}")
            return
    else:
        drone_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('Drone')]
        if not drone_folders:
            print(f"Error: No drone folders found in {base_dir}")
            return
    
    print(f"Found {len(drone_folders)} drone folder(s): {[d.name for d in drone_folders]}")
    total_processed = 0
    
    # Process each drone folder
    for drone_folder in drone_folders:
        print(f"\nProcessing {drone_folder.name}...")
        
        # Find all RGB images
        image_dir = drone_folder / "images"
        if not image_dir.exists():
            print(f"  Warning: Images directory not found in {drone_folder}")
            continue
        
        rgb_files = list(image_dir.glob("img_*_0_*.png"))
        if not rgb_files:
            print(f"  Warning: No RGB images found in {image_dir}")
            continue
        
        print(f"  Found {len(rgb_files)} RGB images")
        
        # Process each RGB image
        processed_count = 0
        for rgb_path in rgb_files:
            # Find corresponding depth file
            depth_path = get_corresponding_depth_file(rgb_path)
            if not depth_path or not depth_path.exists():
                print(f"  Warning: No matching depth file for {rgb_path.name}")
                continue
            
            # Extract timestamp from filename
            timestamp = rgb_path.stem.split('_')[-1]
            
            # Find corresponding model depth file
            model_depth_path = get_model_depth_file(timestamp, drone_folder.name)
            
            # Load RGB image
            rgb_img = Image.open(rgb_path).convert('RGB')
            rgb_np = np.array(rgb_img)
            
            # Load and process depth map
            try:
                depth_data = read_pfm(depth_path)
                # Process depth map
                depth_data = process_single_depth_map(depth_data)
            except Exception as e:
                print(f"  Error reading depth file {depth_path.name}: {e}")
                continue
            
            # Load model depth map if available
            model_depth_data = None
            if model_depth_path and model_depth_path.exists():
                try:
                    model_depth_data = read_pfm(model_depth_path)
                    model_depth_data = process_single_depth_map(model_depth_data)
                except Exception as e:
                    print(f"  Error reading model depth file {model_depth_path.name}: {e}")
            
            # Create figure for visualization
            num_subplots = 3 if model_depth_data is not None else 2
            fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))
            
            # Plot RGB image
            axes[0].imshow(rgb_np)
            axes[0].set_title(f"RGB Image ({drone_folder.name})")
            axes[0].axis('off')
            
            # Plot ground truth depth map
            im = axes[1].imshow(depth_data, cmap='viridis', vmin=0, vmax=200.0)
            axes[1].set_title("Ground Truth Depth")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot model depth if available
            if model_depth_data is not None:
                im_model = axes[2].imshow(model_depth_data, cmap='viridis', vmin=0, vmax=200.0)
                axes[2].set_title("Model Depth Estimation")
                axes[2].axis('off')
                plt.colorbar(im_model, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Add timestamp as title
            plt.suptitle(f"Timestamp: {timestamp}", fontsize=14)
            
            # Save the visualization
            output_filename = f"{drone_folder.name}_timestamp_{timestamp}.png"
            output_path = output_dir / output_filename
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for suptitle
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            processed_count += 1
            total_processed += 1
            
            # Print progress every 10 images
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(rgb_files)} images from {drone_folder.name}")
        
        print(f"  Completed processing {processed_count} images from {drone_folder.name}")
    
    print(f"\nTotal processed: {total_processed} images")
    print(f"Visualizations saved to {output_dir}")

def process_single_depth_map(depth_data):
    """
    Process a single depth map for visualization.
    
    Args:
        depth_data: Numpy array containing depth data
        
    Returns:
        Processed depth data ready for visualization
    """
    # Process depth map
    depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=200.0, neginf=0.0)
    depth_data = np.clip(depth_data, 0, 200.0)
    return depth_data

def visualize_depth_map(depth_data, ax=None, title="Depth Map", with_colorbar=True, vmin=0, vmax=200.0):
    """
    Visualize a single depth map.
    
    Args:
        depth_data: Numpy array containing processed depth data
        ax: Matplotlib axis to plot on. If None, creates a new figure
        title: Title for the plot
        with_colorbar: Whether to add a colorbar
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        The figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Plot depth map
    im = ax.imshow(depth_data, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    
    # Add colorbar if requested
    if with_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax, im

def visualize_rgb_and_depth(rgb_img, depth_data, model_depth_data=None, timestamp=None, output_path=None):
    """
    Create a visualization of an RGB image, ground truth depth, and model depth.
    
    Args:
        rgb_img: RGB image as numpy array [H, W, 3]
        depth_data: Ground truth depth data as numpy array [H, W]
        model_depth_data: Optional model depth data as numpy array [H, W]
        timestamp: Optional timestamp to display
        output_path: Path to save the visualization. If None, just displays the plot
    """
    # Process depth map
    processed_depth = process_single_depth_map(depth_data)
    
    # Determine number of subplots (2 or 3 depending on if model depth is provided)
    num_subplots = 3 if model_depth_data is not None else 2
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))
    
    # Plot RGB image
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')
    
    # Plot ground truth depth map
    _, _, im = visualize_depth_map(processed_depth, ax=axes[1], title="Ground Truth Depth")
    
    # Plot model depth if provided
    if model_depth_data is not None:
        processed_model_depth = process_single_depth_map(model_depth_data)
        _, _, _ = visualize_depth_map(processed_model_depth, ax=axes[2], title="Model Depth Estimation")
    
    # Add timestamp as title if provided
    if timestamp is not None:
        plt.suptitle(f"Timestamp: {timestamp}", fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        return fig, axes

def process_single_image(image_path):
    """
    Process a single RGB image and its corresponding depth map.
    
    Args:
        image_path: Path to the RGB image
    """
    rgb_path = Path(image_path)
    
    if not rgb_path.exists():
        print(f"Error: Image file {image_path} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path("depth_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract timestamp and drone ID from filename if possible
    filename_parts = rgb_path.stem.split('_')
    timestamp = filename_parts[-1] if len(filename_parts) >= 4 else "unknown"
    drone_id = filename_parts[1] if len(filename_parts) >= 2 and filename_parts[0] == "img" else "unknown"
    
    # Find corresponding depth file
    depth_path = get_corresponding_depth_file(rgb_path)
    if not depth_path or not depth_path.exists():
        print(f"Warning: No matching depth file for {rgb_path.name}")
        return
    
    # Find corresponding model depth file
    model_depth_path = get_model_depth_file(timestamp, drone_id)
    
    # Load RGB image
    rgb_img = Image.open(rgb_path).convert('RGB')
    rgb_np = np.array(rgb_img)
    
    # Load and process depth map
    try:
        depth_data = read_pfm(depth_path)
        # Process depth map
        depth_data = process_single_depth_map(depth_data)
    except Exception as e:
        print(f"Error reading depth file {depth_path.name}: {e}")
        return
    
    # Load model depth map if available
    model_depth_data = None
    if model_depth_path and model_depth_path.exists():
        try:
            model_depth_data = read_pfm(model_depth_path)
        except Exception as e:
            print(f"Error reading model depth file {model_depth_path.name}: {e}")
    else:
        print(f"No model depth file found for timestamp {timestamp}, drone {drone_id}")
    
    # Create figure for visualization
    num_subplots = 3 if model_depth_data is not None else 2
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))
    
    # Plot RGB image
    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB Image ({drone_id})")
    axes[0].axis('off')
    
    # Plot ground truth depth map
    im = axes[1].imshow(depth_data, cmap='viridis', vmin=0, vmax=200.0)
    axes[1].set_title("Ground Truth Depth")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot model depth if available
    if model_depth_data is not None:
        model_depth_data = process_single_depth_map(model_depth_data)
        im_model = axes[2].imshow(model_depth_data, cmap='viridis', vmin=0, vmax=200.0)
        axes[2].set_title("Model Depth Estimation")
        axes[2].axis('off')
        plt.colorbar(im_model, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add timestamp as title
    plt.suptitle(f"Timestamp: {timestamp}", fontsize=14)
    
    # Save the visualization
    output_filename = f"single_image_{drone_id}_timestamp_{timestamp}.png"
    output_path = output_dir / output_filename
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize RGB images and their corresponding depth maps')
    parser.add_argument('--base_dir', type=str, default="training/four_drones", 
                        help='Base directory containing drone folders')
    parser.add_argument('--drone_id', type=str, default=None,
                        help='Specific drone ID to process (e.g., "Drone0"). If not specified, process all drones.')
    
    # Single image processing option
    parser.add_argument('--single_image', type=str, default=None,
                        help='Path to a single RGB image to process (instead of processing folders)')
    
    args = parser.parse_args()
    
    # Process a single image if specified
    if args.single_image:
        process_single_image(args.single_image)
    else:
        # Process the entire directory structure
        print(f"Processing drone data from: {args.base_dir}")
        process_drone_folder(args.base_dir, args.drone_id)
    
    print("Done!")

if __name__ == "__main__":
    main()
