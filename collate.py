import torch

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
