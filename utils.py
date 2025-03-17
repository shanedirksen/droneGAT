# c:\Users\shane\Documents\GitHub\AirSim\drone_gat\utils.py
import torch
import numpy as np

def build_graph_from_positions(positions, distance_threshold):
    """
    Build graph connectivity based on drone positions
    Args:
        positions: [num_drones, 3] tensor of (x, y, z) coordinates
        distance_threshold: maximum distance to create an edge
    Returns:
        edge_index: [2, num_edges] tensor defining graph connectivity
    """
    num_drones = positions.shape[0]
    edges = []
    
    # Connect drones within the distance threshold
    for i in range(num_drones):
        for j in range(num_drones):
            if i != j:
                dist = torch.norm(positions[i] - positions[j])
                if dist <= distance_threshold:
                    edges.append([i, j])
    
    if not edges:
        # Return empty edge index tensor on the same device as positions
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        
    edge_index = torch.tensor(edges, dtype=torch.long, device=positions.device).t()
    return edge_index

def generate_spatial_encoding(positions, grid_size=32, max_range=100.0):
    """
    Generate spatial encoding for drone positions
    Args:
        positions: [num_drones, 3] tensor of (x, y, z) coordinates
        grid_size: size of the grid for spatial encoding
        max_range: maximum range for position normalization
    Returns:
        encoding: [num_drones, 3, grid_size, grid_size] spatial encoding tensor
    """
    num_drones = positions.shape[0]
    device = positions.device
    encoding = torch.zeros(num_drones, 3, grid_size, grid_size, device=device)
    
    # Normalize positions
    norm_pos = positions.clone() / max_range
    norm_pos = torch.clamp(norm_pos, 0, 1)
    
    for i in range(num_drones):
        # Create spatial encoding with Gaussian peaks - ensure all tensors are on the same device
        x_grid = torch.linspace(0, 1, grid_size, device=device).unsqueeze(0).repeat(grid_size, 1)
        y_grid = torch.linspace(0, 1, grid_size, device=device).unsqueeze(1).repeat(1, grid_size)
        
        # X channel - Gaussian based on x-position
        encoding[i, 0] = torch.exp(-((x_grid - norm_pos[i, 0]) ** 2) / 0.1)
        
        # Y channel - Gaussian based on y-position
        encoding[i, 1] = torch.exp(-((y_grid - norm_pos[i, 1]) ** 2) / 0.1)
        
        # Z channel - uniform value based on z-position
        encoding[i, 2] = norm_pos[i, 2]
    
    return encoding

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        q: quaternion tensor [w, x, y, z]
        
    Returns:
        roll, pitch, yaw angles in radians
    """
    # Extract quaternion components
    qw, qx, qy, qz = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    # Handle gimbal lock edge cases
    if torch.abs(sinp) >= 1:
        pitch = torch.sign(sinp) * torch.pi / 2.0  # use 90 degrees if out of range
    else:
        pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw])

def calculate_relative_poses(positions, orientations, edge_index):
    """
    Calculate relative poses between drones based on edge connections.
    
    Args:
        positions: Tensor of shape [num_drones, 3] with drone positions (x, y, z)
        orientations: Tensor of shape [num_drones, 4] with drone orientations as quaternions (w, x, y, z)
        edge_index: Graph connectivity tensor of shape [2, num_edges]
    
    Returns:
        Tensor of shape [num_edges, 6] with relative poses (dx, dy, dz, droll, dpitch, dyaw)
    """
    device = positions.device
    num_edges = edge_index.shape[1]
    
    if num_edges == 0:
        return torch.zeros((0, 6), device=device)
    
    # Initialize tensor to store relative poses
    relative_poses = torch.zeros((num_edges, 6), device=device)
    
    for e in range(num_edges):
        # Get source and target drone indices
        source_idx = edge_index[0, e]
        target_idx = edge_index[1, e]
        
        # Calculate relative position (target position in source frame)
        rel_position = positions[target_idx] - positions[source_idx]
        
        # Convert quaternions to Euler angles
        source_euler = quaternion_to_euler(orientations[source_idx])
        target_euler = quaternion_to_euler(orientations[target_idx])
        
        # Calculate relative orientation (simple difference for now)
        # For a more accurate calculation, you might want to use proper rotation composition
        rel_orientation = target_euler - source_euler
        
        # Normalize angles to [-π, π]
        rel_orientation = torch.remainder(rel_orientation + torch.pi, 2 * torch.pi) - torch.pi
        
        # Store relative pose
        relative_poses[e, :3] = rel_position
        relative_poses[e, 3:] = rel_orientation
    
    return relative_poses

def visualize_distance_matrix(positions, distance_threshold, timestamp=None, save_path=None):
    """
    Create and save a visualization of the distance matrix between drones
    
    Args:
        positions: Numpy array or tensor of shape [num_drones, 3] with drone positions
        distance_threshold: Distance threshold for edge creation
        timestamp: Optional timestamp for the title
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy if tensor
    if isinstance(positions, torch.Tensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = positions
        
    num_drones = positions_np.shape[0]
    
    # Create a distance matrix
    distances = np.zeros((num_drones, num_drones))
    for i in range(num_drones):
        for j in range(num_drones):
            if i != j:
                distances[i, j] = np.linalg.norm(positions_np[i] - positions_np[j])
            else:
                distances[i, j] = 0.0
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
    ax.axis('off')
    
    # Create table with drone names as row and column headers
    drone_names = [f"Drone{i}" for i in range(num_drones)]
    
    # Format distances with two decimal places
    cell_text = [[f"{d:.2f}" for d in row] for row in distances]
    
    # Add a title with the timestamp if provided
    title = "Drone Distance Matrix"
    if timestamp is not None:
        title += f" (Timestamp: {timestamp:.2f})"
    plt.title(title, fontsize=14)
    plt.figtext(0.5, 0.01, f"Distance threshold for edge creation: {distance_threshold}", 
               ha='center', fontsize=12)
    
    # Color cells where distance is below threshold (will have edges)
    cell_colors = np.zeros((num_drones, num_drones, 4))  # RGBA
    for i in range(num_drones):
        for j in range(num_drones):
            if i != j and distances[i, j] <= distance_threshold:
                # Light green for distances below threshold
                cell_colors[i, j] = [0.8, 1.0, 0.8, 1.0]
            elif i == j:
                # Light gray for diagonal
                cell_colors[i, j] = [0.9, 0.9, 0.9, 1.0]
            else:
                # White for distances above threshold
                cell_colors[i, j] = [1.0, 1.0, 1.0, 1.0]
    
    # Create table
    table = plt.table(cellText=cell_text,
                     rowLabels=drone_names,
                     colLabels=drone_names,
                     cellColours=cell_colors,
                     loc='center')
    
    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Save the figure if a path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        return fig

def visualize_3d_graph(positions, edge_index, distance_threshold, timestamp=None, save_path=None):
    """
    Create and save a 3D visualization of the drone communication graph
    
    Args:
        positions: Tensor or numpy array of shape [num_drones, 3] with drone positions
        edge_index: Graph connectivity tensor of shape [2, num_edges]
        distance_threshold: Distance threshold used for edge creation
        timestamp: Optional timestamp for the title
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Convert to numpy if tensor
    if isinstance(positions, torch.Tensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = positions
        
    if isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.cpu().numpy()
    else:
        edge_index_np = edge_index
    
    # Create a figure for the 3D graph
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot drone positions as nodes
    ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], 
              c='blue', marker='o', s=100, label='Drones')
    
    # Add drone ID labels
    for j, pos in enumerate(positions_np):
        ax.text(pos[0], pos[1], pos[2], f'Drone{j}', fontsize=10)
    
    # If there are edges, plot them
    if edge_index_np.size > 0 and edge_index_np.shape[1] > 0:
        edges = edge_index_np.T if edge_index_np.shape[0] == 2 else edge_index_np
        
        # Plot edges
        for edge in edges:
            src, dst = edge
            ax.plot([positions_np[src, 0], positions_np[dst, 0]],
                   [positions_np[src, 1], positions_np[dst, 1]],
                   [positions_np[src, 2], positions_np[dst, 2]],
                   'r-', alpha=0.7, linewidth=1.5)
    
    # Set plot labels and limits
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    # Set title with timestamp if provided
    title = 'Drone Communication Graph'
    if timestamp is not None:
        title += f' (Timestamp: {timestamp:.2f})'
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Enhance the 3D view
    ax.view_init(elev=30, azim=45)
    
    # Add distance threshold info
    plt.figtext(0.02, 0.02, f"Distance Threshold: {distance_threshold}", fontsize=10)
    
    # Add graph statistics
    num_edges = edge_index_np.shape[1] if edge_index_np.size > 0 and edge_index_np.shape[0] == 2 else 0
    num_nodes = positions_np.shape[0]
    plt.figtext(0.02, 0.05, f"Nodes: {num_nodes}, Edges: {num_edges}", fontsize=10)
    
    # Save the figure if a path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        return fig

def visualize_2d_graph(positions, edge_index, distance_threshold, timestamp=None, save_path=None):
    """
    Create and save a 2D NetworkX visualization of the drone communication graph
    
    Args:
        positions: Tensor or numpy array of shape [num_drones, 3] with drone positions
        edge_index: Graph connectivity tensor of shape [2, num_edges]
        distance_threshold: Distance threshold used for edge creation
        timestamp: Optional timestamp for the title
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # Convert to numpy if tensor
    if isinstance(positions, torch.Tensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = positions
        
    if isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.cpu().numpy()
    else:
        edge_index_np = edge_index
    
    plt.figure(figsize=(10, 8))
    
    G = nx.DiGraph()
    
    # Add nodes
    for j in range(positions_np.shape[0]):
        G.add_node(j, pos=(positions_np[j, 0], positions_np[j, 1]))
        
    # Add edges
    if edge_index_np.size > 0 and edge_index_np.shape[1] > 0:
        for e in range(edge_index_np.shape[1]):
            src = int(edge_index_np[0, e])
            dst = int(edge_index_np[1, e])
            G.add_edge(src, dst)
    
    # Draw nodes with labels
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
           node_size=500, font_size=12, font_weight='bold',
           arrowsize=15)
    
    # Set title with timestamp if provided
    title = '2D Drone Communication Graph'
    if timestamp is not None:
        title += f' (Timestamp: {timestamp:.2f})'
    plt.title(title)
    
    # Add stats text box
    num_nodes = positions_np.shape[0]
    num_edges = edge_index_np.shape[1] if edge_index_np.size > 0 and edge_index_np.shape[0] == 2 else 0
    plt.figtext(0.05, 0.05, f"Nodes: {num_nodes}\nEdges: {num_edges}\nThreshold: {distance_threshold}", 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the figure if a path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()

def compute_depth_metrics(pred, target, mask=None):
    """
    Compute standard depth estimation metrics
    
    Args:
        pred: Predicted depth [B, 1, H, W]
        target: Ground truth depth [B, 1, H, W]
        mask: Optional mask for valid depth regions [B, 1, H, W]
    
    Returns:
        Dictionary of metrics:
            - abs_rel: Absolute Relative difference (lower is better)
            - sq_rel: Squared Relative difference (lower is better)
            - rmse: Root Mean Squared Error (lower is better)
    """
    import torch
    metrics = {}
    
    # If we don't have a mask, create one for all non-zero values
    if mask is None:
        mask = (target > 0).float()
    
    # Make sure we're working with tensors
    pred = torch.as_tensor(pred)
    target = torch.as_tensor(target)
    
    # Apply mask
    pred_masked = pred * mask
    target_masked = target * mask
    
    # Count valid pixels
    num_valid_pixels = mask.sum()
    
    if num_valid_pixels == 0:
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan')
        }
    
    # Compute metrics
    # Absolute Relative Difference
    abs_diff = torch.abs(pred_masked - target_masked)
    abs_rel = torch.sum(abs_diff / (target_masked + 1e-10)) / num_valid_pixels
    
    # Squared Relative Difference
    sq_rel = torch.sum(torch.pow(abs_diff, 2) / (target_masked + 1e-10)) / num_valid_pixels
    
    # RMSE
    rmse = torch.sqrt(torch.sum(torch.pow(abs_diff, 2)) / num_valid_pixels)
    
    metrics['abs_rel'] = abs_rel.item()
    metrics['sq_rel'] = sq_rel.item()
    metrics['rmse'] = rmse.item()
    
    return metrics

def compute_segmentation_metrics(pred, target, num_classes=3):
    """
    Compute segmentation metrics
    
    Args:
        pred: Predicted segmentation logits [B, C, H, W]
        target: Ground truth segmentation [B, H, W] with class indices
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics:
            - miou: Mean Intersection over Union (higher is better)
            - pixel_acc: Pixel Accuracy (higher is better)
    """
    import torch
    metrics = {}
    
    # Convert logits to class indices
    if pred.dim() == 4 and pred.size(1) > 1:
        pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    # If target is one-hot, convert to indices
    if target.dim() == 4 and target.size(1) > 1:
        target = torch.argmax(target, dim=1)  # [B, H, W]
        
    # Ensure correct shape
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)  # [B, H, W]
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)  # [B, H, W]
    
    # Compute IoU for each class
    iou_sum = 0.0
    valid_classes = 0
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            iou_sum += iou.item()
            valid_classes += 1
    
    # Compute mean IoU
    if valid_classes > 0:
        miou = iou_sum / valid_classes
    else:
        miou = 0.0
    
    # Compute pixel accuracy
    correct = (pred == target).sum().float()
    total = pred.numel()
    pixel_acc = correct / total
    
    metrics['miou'] = miou
    metrics['pixel_acc'] = pixel_acc.item()
    
    return metrics

def prepare_batch_for_model(batch, device, distance_threshold):
    """
    Prepare batch data for model input
    
    Args:
        batch: Dictionary containing batch data
        device: Device to move tensors to
        distance_threshold: Threshold for building graph connections
        
    Returns:
        Tuple of (rgb_features, edge_indices, rel_poses, targets) for model input
    """
    rgb_features = []
    edge_indices = []
    rel_poses = []
    targets = {'depth': [], 'segmentation': []}
    
    for i, (rgb, positions, orientations) in enumerate(zip(batch['rgb'], batch['positions'], batch['orientations'])):
        # Move tensors to device
        rgb = rgb.to(device)
        positions = positions.to(device)
        orientations = orientations.to(device)
        
        # Generate edge index (graph connectivity)
        edge_index = build_graph_from_positions(positions, distance_threshold)
        
        # Calculate relative poses between drones if edges exist
        if edge_index.size(1) > 0:
            rel_pose = calculate_relative_poses(positions, orientations, edge_index)
            rel_poses.append(rel_pose)
        else:
            rel_poses.append(None)
        
        # Add to batch data
        rgb_features.append(rgb)  # Shape: [num_drones, 3, H, W]
        edge_indices.append(edge_index)
        
        # Add targets if available
        if 'depth' in batch and i < len(batch['depth']):
            targets['depth'].append(batch['depth'][i].to(device))
        if 'segmentation' in batch and i < len(batch['segmentation']):
            targets['segmentation'].append(batch['segmentation'][i].to(device))
    
    return rgb_features, edge_indices, rel_poses, targets