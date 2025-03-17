# c:\Users\shane\Documents\GitHub\AirSim\drone_gat\data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import re
from pathlib import Path
import glob
import random

def read_pfm(file):
    """Read PFM depth file and return as numpy array."""
    with open(file, 'rb') as f:
        # Header
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        
        # Dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
            
        # Scale
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'  # little-endian or big-endian
            
        # Data
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        
        # REMOVED the vertical flip here to avoid duplicating with depth_visualization.py
        # Let only depth_visualization.py handle the flipping
        
        # Create a contiguous copy to avoid negative stride issues with PyTorch
        data = data.copy()
        
    return data

class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None, corruption_type='both', 
                 random_corruption=False, max_corrupt_ratio=0.5, mode='train', test_corruption_percentage=33):
        """
        Args:
            root_dir (string): Directory with all the drone data.
            transform (callable): Optional transform to be applied on RGB images.
            corruption_type (string): Type of corruption ('motion_blur', 'shot_noise', 'both', None).
            random_corruption (bool): If True, randomly corrupt some drones during training.
            max_corrupt_ratio (float): Maximum ratio of drones that can be corrupted (0.0-1.0).
            mode (str): 'train', 'val', or 'test' - controls corruption behavior.
            test_corruption_percentage (int): Percentage of drones to corrupt during testing (0, 33, 100).
        """
        self.root_dir = Path(root_dir)
        self.rgb_transform = transform  # Transform for RGB images
        self.corruption_type = corruption_type
        self.random_corruption = random_corruption
        self.max_corrupt_ratio = max_corrupt_ratio
        self.mode = mode
        self.test_corruption_percentage = test_corruption_percentage
        
        # Valid corruption types and intensities
        self.valid_corruptions = ['motion_blur', 'shot_noise']
        self.valid_intensities = [1, 3, 5]
        
        # Auto-detect the number of drones by counting drone directories
        drone_dirs = [d for d in self.root_dir.glob("Drone*") if d.is_dir()]
        self.num_drones = len(drone_dirs)
        
        if self.num_drones == 0:
            raise ValueError(f"No drone directories found in {root_dir}. Expected directories named 'Drone0', 'Drone1', etc.")
        
        print(f"Found {self.num_drones} drones in the dataset directory")
        
        # Create separate transforms for RGB, depth, and segmentation
        if transform:
            from torchvision import transforms
            # Original RGB transform remains unchanged
            self.rgb_transform = transform
            
            # Direct numpy-based resize for depth maps
            self.depth_transform = self.depth_transform_np
            
            # Segmentation-specific transform that preserves class values
            self.seg_transform = self.segmentation_transform_np
        else:
            self.rgb_transform = None
            self.depth_transform = None
            self.seg_transform = None
        
        # Define color-to-class mapping for segmentation
        # This is a placeholder - update with actual color mappings from your dataset
        self.segmentation_colors = {
            # Example format: RGB tuple -> class index
            (0, 0, 0): 0,      # Background - black
            (128, 128, 128): 1,  # Buildings - gray
            (0, 0, 255): 2,      # Road - blue
            (0, 255, 0): 3,      # Vegetation - green
            (255, 0, 0): 4,      # Vehicle - red
            # Add more classes as needed
        }
        
        self.drone_data = []
        
        # Load data for each drone - using auto-detected number of drones
        for drone_id in range(0, self.num_drones):
            drone_path = self.root_dir / f"Drone{drone_id}"
            
            # Parse airsim_rec.txt file
            airsim_rec_path = drone_path / "airsim_rec.txt"
            records = self._parse_airsim_rec(airsim_rec_path)
            
            # Store records for this drone
            self.drone_data.append(records)
        
        # Find common timestamps across all drones
        self.timestamps = self._find_common_timestamps()
        
        # Verify corruption directories exist
        self._verify_corruption_dirs()
    
    def _verify_corruption_dirs(self):
        """Verify that corruption directories exist for at least one drone."""
        if not self.corruption_type and not self.random_corruption:
            return
            
        sample_drone_dir = self.root_dir / "Drone0"
        corruption_dirs_exist = False
        
        for corruption_type in self.valid_corruptions:
            for intensity in self.valid_intensities:
                corruption_dir = sample_drone_dir / f"corrupt_{corruption_type}_{intensity}"
                if corruption_dir.exists() and corruption_dir.is_dir():
                    corruption_dirs_exist = True
                    break
            if corruption_dirs_exist:
                break
                
        if not corruption_dirs_exist:
            print(f"Warning: No corruption directories found in {sample_drone_dir}. "
                  f"Expected directories like 'corrupt_motion_blur_1', etc.")

    def _parse_airsim_rec(self, filepath):
        """Parse the airsim_rec.txt file and return records."""
        records = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("# ") or "VehicleName" in line:  # Skip header lines
                    continue
                    
                parts = line.strip().split("\t")
                if len(parts) >= 9:  # Ensure we have all required fields
                    try:
                        # The format is: Drone_ID timestamp pos_x pos_y pos_z q_w q_x q_y q_z image_paths
                        # Extract the drone_id (already handled by our outer loop)
                        timestamp = float(parts[1])
                        pos_x, pos_y, pos_z = map(float, parts[2:5])
                        q_w, q_x, q_y, q_z = map(float, parts[5:9])
                        
                        # Extract image filenames - they're in the last column separated by semicolons
                        image_paths = parts[-1].split(';')
                        
                        # Get each type of image path
                        rgb_path = None
                        depth_path = None
                        seg_path = None
                        
                        for path in image_paths:
                            if '_0_' in path:  # RGB image
                                rgb_path = path
                            elif '_1_' in path:  # Depth image
                                depth_path = path
                            elif '_5_' in path:  # Segmentation image
                                seg_path = path
                        
                        # Create record
                        record = {
                            'timestamp': timestamp,
                            'position': torch.tensor([pos_x, pos_y, pos_z]),
                            'orientation': torch.tensor([q_w, q_x, q_y, q_z]),
                            'rgb_path': rgb_path,
                            'depth_path': depth_path,
                            'seg_path': seg_path
                        }
                        records.append(record)
                    except ValueError as e:
                        # Skip lines that cannot be parsed properly
                        print(f"Warning: Skipping line in airsim_rec.txt: {line.strip()}")
                        continue
        
        return records
    
    def _find_common_timestamps(self):
        """Find timestamps that exist for all drones."""
        if not self.drone_data:
            return []
            
        # Extract timestamps for each drone
        drone_timestamps = []
        for drone_records in self.drone_data:
            timestamps = [record['timestamp'] for record in drone_records]
            drone_timestamps.append(set(timestamps))
        
        # Find common timestamps
        common_timestamps = drone_timestamps[0]
        for ts_set in drone_timestamps[1:]:
            common_timestamps = common_timestamps.intersection(ts_set)
        
        return sorted(list(common_timestamps))
    
    def _get_image_path(self, file_path, drone_id, corruption_type=None, force_corruption=False):
        """
        Get image path based on corruption settings.
        
        Args:
            file_path: Original file path
            drone_id: ID of the drone
            corruption_type: Type of corruption to apply ('motion_blur', 'shot_noise', 'both', None)
            force_corruption: If True, always use corrupted image if available
            
        Returns:
            Path to the image file to use
        """
        if not file_path:
            return None
            
        path = Path(file_path)
        
        # If no corruption or force_corruption is False, return the original path
        if corruption_type is None and not force_corruption:
            return path
        
        # For corrupted images, construct the path based on corruption type and intensity
        filename = path.name
        drone_dir = self.root_dir / f"Drone{drone_id}"
        
        # If corruption_type is 'both' or force_corruption is True and no type specified,
        # randomly select one of the corruption types
        selected_type = corruption_type
        
        if corruption_type == 'both' or (corruption_type is None and force_corruption):
            selected_type = random.choice(self.valid_corruptions)
            
        # Always randomly select an intensity level from the valid options
        selected_intensity = random.choice(self.valid_intensities)
            
        # Construct the corruption path
        corruption_folder = f"corrupt_{selected_type}_{selected_intensity}"
        corruption_path = drone_dir / corruption_folder / filename
        
        # Return the corruption path if it exists, otherwise return original path
        return corruption_path if corruption_path.exists() else path
    
    def depth_transform_np(self, depth_tensor):
        """
        Transform depth data using direct numpy operations (no PIL)
        
        Args:
            depth_tensor: Depth tensor with shape [C, H, W] or [H, W]
        
        Returns:
            Resized depth tensor with shape [C, 224, 224] or [1, 224, 224]
        """
        # Convert to numpy if tensor
        if isinstance(depth_tensor, torch.Tensor):
            if depth_tensor.dim() == 3:  # [C, H, W]
                depth_np = depth_tensor.numpy()[0]  # Take first channel
            else:  # [H, W]
                depth_np = depth_tensor.numpy()
        else:
            depth_np = depth_tensor
            
        # Resize directly using OpenCV
        depth_resized = cv2.resize(depth_np, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to tensor and add channel dimension if needed
        depth_tensor = torch.from_numpy(depth_resized).float()
        if depth_tensor.dim() == 2:  # [H, W]
            depth_tensor = depth_tensor.unsqueeze(0)  # Make [1, H, W]
            
        return depth_tensor
    
    def segmentation_transform_np(self, seg_tensor):
        """
        Transform colored segmentation mask using nearest-neighbor interpolation 
        to preserve exact colors
        
        Args:
            seg_tensor: Segmentation tensor with shape [C, H, W] or [H, W, C]
        
        Returns:
            Resized segmentation tensor with shape [C, 224, 224]
        """
        # Convert to numpy if tensor
        if isinstance(seg_tensor, torch.Tensor):
            if seg_tensor.dim() == 3 and seg_tensor.size(0) == 3:  # [3, H, W] - channels first
                seg_np = seg_tensor.permute(1, 2, 0).numpy()  # Convert to [H, W, 3] for OpenCV
            elif seg_tensor.dim() == 3 and seg_tensor.size(2) == 3:  # [H, W, 3] - already correct
                seg_np = seg_tensor.numpy()
            else:  # Unexpected shape
                raise ValueError(f"Unexpected segmentation tensor shape: {seg_tensor.shape}")
        else:
            seg_np = seg_tensor
            
        # Resize using NEAREST interpolation to preserve exact colors
        seg_resized = cv2.resize(seg_np, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Convert back to tensor in PyTorch's preferred format [C, H, W]
        seg_tensor = torch.from_numpy(seg_resized).float().permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            
        return seg_tensor
    
    def _load_image(self, image_path, image_type):
        """Load image based on type."""
        if image_path is None:
            return None
            
        # Convert to Path object if it's a string
        if isinstance(image_path, str):
            image_path = Path(image_path)
            
        if not image_path.exists():
            return None
            
        if image_type == 1:  # Depth
            # Load depth data with the consistent read_pfm function - No changes here
            depth_data = read_pfm(image_path)
            
            # Convert to proper format for model - no normalization, just convert to tensor
            depth_tensor = torch.from_numpy(depth_data).float().unsqueeze(0)  # Add channel dimension [1, H, W]
            return depth_tensor
        elif image_type == 5:  # Segmentation - load as color
            # Load the segmentation mask with OpenCV as color image
            seg_mask = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if seg_mask is None:
                return None
                
            # Convert BGR to RGB (OpenCV loads as BGR)
            seg_mask_rgb = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor [H, W, 3] -> [3, H, W]
            seg_tensor = torch.from_numpy(seg_mask_rgb).float().permute(2, 0, 1)
            return seg_tensor
        else:  # RGB
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB if it's an RGB image
            if image_type == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to tensor
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            # Scale to [0, 1] range
            img_tensor = img_tensor / 255.0
            
            return img_tensor
    
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        timestamp = self.timestamps[idx]
        
        # Collect data from all drones at this timestamp
        drone_features = []
        drone_positions = []
        
        # Determine which drones to corrupt based on mode and settings
        corrupt_drones = set()
        
        # Apply corruption logic based on mode
        if self.mode == 'train' and self.random_corruption:
            # During training with random corruption:
            # Randomly select up to max_corrupt_ratio of drones to corrupt
            max_corrupt = max(1, int(self.num_drones * self.max_corrupt_ratio))
            num_corrupt = random.randint(0, max_corrupt)
            corrupt_drones = set(random.sample(range(self.num_drones), num_corrupt))
            
        elif self.mode == 'test':
            if self.corruption_type is not None:
                if self.test_corruption_percentage == 100:
                    # 100% corruption: corrupt all drones
                    corrupt_drones = set(range(self.num_drones))
                elif self.test_corruption_percentage == 33:
                    # 33% corruption: corrupt about 1/3 of drones
                    num_corrupt = max(1, int(self.num_drones * 0.33))
                    corrupt_drones = set(random.sample(range(self.num_drones), num_corrupt))
                else:
                    # 0% corruption: don't corrupt any drones
                    corrupt_drones = set()
        
        # Change to use zero-indexed drone IDs
        for drone_id, drone_records in enumerate(self.drone_data):
            # Find the record with the closest timestamp
            closest_record = None
            min_diff = float('inf')
            
            for record in drone_records:
                diff = abs(record['timestamp'] - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_record = record
            
            if closest_record:
                # Determine if this drone should use corrupted images
                use_corruption = drone_id in corrupt_drones
                
                # Get image paths with corruption applied if needed
                if use_corruption:
                    # Force corruption for this drone with randomly selected intensity
                    rgb_path = self._get_image_path(
                        closest_record['rgb_path'],
                        drone_id,
                        self.corruption_type,  # Can be 'both', will be randomly selected if needed
                        force_corruption=True
                    )
                else:
                    # Use default corruption settings (might be None)
                    rgb_path = self._get_image_path(
                        closest_record['rgb_path'],
                        drone_id,
                        None
                    )
                
                # For depth and segmentation, always use the original paths
                depth_path = self._get_image_path(closest_record['depth_path'], drone_id)
                seg_path = self._get_image_path(closest_record['seg_path'], drone_id)
                
                # Load images
                rgb_img = self._load_image(rgb_path, 0)
                depth_img = self._load_image(depth_path, 1)
                seg_img = self._load_image(seg_path, 5)
                
                # Apply transforms if provided - use appropriate transform for each image type
                if self.rgb_transform:
                    if rgb_img is not None:
                        rgb_img = self.rgb_transform(rgb_img)
                    if depth_img is not None:
                        # Use direct numpy transform for depth, avoiding PIL conversions
                        if self.depth_transform:
                            depth_img = self.depth_transform(depth_img)
                    if seg_img is not None:
                        # Use segmentation-specific transform with nearest neighbor interpolation
                        # to preserve color values
                        if self.seg_transform:
                            seg_img = self.seg_transform(seg_img)
                
                # Store position for creating graph adjacency
                drone_positions.append(closest_record['position'])
                
                # Create feature tensor for this drone
                features = {
                    'drone_id': drone_id,
                    'timestamp': timestamp,
                    'position': closest_record['position'],
                    'orientation': closest_record['orientation'],
                    'rgb': rgb_img,
                    'depth': depth_img,
                    'segmentation': seg_img,
                    'corrupted': use_corruption  # Add flag to indicate if this drone used corrupted image
                }
                
                drone_features.append(features)
        
        # Create adjacency matrix for GAT based on drone positions
        positions = torch.stack(drone_positions)
        distances = torch.cdist(positions, positions)
        
        # Create adjacency matrix - can be modified based on desired connectivity
        adjacency = torch.ones(self.num_drones, self.num_drones)
        
        # Remove self-loops
        for i in range(self.num_drones):
            adjacency[i, i] = 0
        
        # Count number of corrupted drones for logging/debugging
        num_corrupted = sum(1 for feature in drone_features if feature.get('corrupted', False))
        
        return {
            'drone_features': drone_features,
            'adjacency': adjacency,
            'positions': positions,
            'timestamp': timestamp,
            'num_corrupted': num_corrupted
        }

def create_drone_dataloader(root_dir, batch_size=1, transform=None, 
                           corruption_type='both', shuffle=True,
                           num_workers=4, random_corruption=False, max_corrupt_ratio=0.5,
                           mode='train', test_corruption_percentage=33):
    """
    Create a DataLoader for the drone dataset.
    
    Args:
        root_dir: Root directory containing drone data
        batch_size: Batch size for training
        transform: Transforms to apply to images
        corruption_type: Type of corruption ('motion_blur', 'shot_noise', 'both', None)
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        random_corruption: If True, randomly corrupt some drones during training
        max_corrupt_ratio: Maximum ratio of drones that can be corrupted (0.0-1.0)
        mode: 'train', 'val', or 'test' - controls corruption behavior
        test_corruption_percentage: Percentage of drones to corrupt during testing (0, 33, 100)
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = DroneDataset(
        root_dir=root_dir,
        transform=transform,
        corruption_type=corruption_type,
        random_corruption=random_corruption,
        max_corrupt_ratio=max_corrupt_ratio,
        mode=mode,
        test_corruption_percentage=test_corruption_percentage
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )