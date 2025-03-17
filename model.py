# c:\Users\shane\Documents\GitHub\AirSim\drone_gat\model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class DroneFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(DroneFeatureExtractor, self).__init__()
        
        # Use MobileNetV2 as the backbone feature extractor as specified in the paper
        if pretrained:
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            self.backbone = mobilenet_v2(weights=None)
        
        # Modify first conv layer if input channels != 3 (e.g., for depth or segmentation inputs)
        if input_channels != 3:
            # Save weights and bias of the original first conv layer
            original_conv = self.backbone.features[0][0]
            original_weight = original_conv.weight.data
            original_bias = original_conv.bias.data if original_conv.bias is not None else None
            
            # Create a new conv layer with the desired input channels
            new_conv = nn.Conv2d(
                input_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize the new layer with the original weights
            if input_channels <= 3:
                new_conv.weight.data[:, :input_channels] = original_weight[:, :input_channels]
            else:
                # Repeat the original weights for additional channels
                new_conv.weight.data[:, :3] = original_weight
                for i in range(3, input_channels):
                    new_conv.weight.data[:, i] = original_weight[:, i % 3]
            
            if original_bias is not None:
                new_conv.bias.data = original_bias
                
            # Replace the first conv layer
            self.backbone.features[0][0] = new_conv
        
        # Remove the classifier head
        self.features = self.backbone.features
        
        # Feature dimensions
        self.feature_channels = 1280  # MobileNetV2 last layer has 1280 channels
        self.feature_size = (7, 7)  # Default size for 224x224 input
        
    def forward(self, x):
        # Extract features from all layers excluding the classifier
        features = self.features(x)
        return features  # Output shape: [batch_size, 1280, h, w]

class SpatialEncoding(nn.Module):
    """Encodes the relative pose between robots as described in the paper"""
    def __init__(self, feature_dim, hidden_dim):
        super(SpatialEncoding, self).__init__()
        
        # Encode relative pose information (position + orientation)
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim//2),  # Position (3) + Rotation (3) = 6
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # FiLM-like feature modulation layers
        self.gamma = nn.Linear(hidden_dim, feature_dim)
        self.beta = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, features, rel_pose):
        """
        Apply spatial encoding to features based on relative pose
        
        Args:
            features: Feature vector of target robot [feature_dim]
            rel_pose: Relative pose from source to target [6] (x,y,z,roll,pitch,yaw)
                     or batch of poses [batch_size, 6]
        """
        batch_mode = (len(rel_pose.shape) == 2)
        
        # Encode relative pose
        pose_features = self.pose_encoder(rel_pose)
        
        # Generate modulation parameters (FiLM)
        gamma = torch.sigmoid(self.gamma(pose_features)) * 2  # Scale in [0,2]
        beta = self.beta(pose_features)
        
        # Apply feature modulation: gamma * features + beta
        if batch_mode:
            # For batch processing: [batch_size, feature_dim] * [batch_size, feature_dim] + [batch_size, feature_dim]
            modulated_features = gamma * features + beta
        else:
            # For single feature processing
            modulated_features = gamma * features + beta
        
        return modulated_features

class DroneGAT(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128, output_dim=64, 
                 num_heads=8, dropout=0.1, pretrained=True, 
                 predict_depth=True, predict_segmentation=True):
        super(DroneGAT, self).__init__()
        
        # Feature extraction with MobileNetV2
        self.feature_extractor = DroneFeatureExtractor(input_channels, pretrained)
        feature_dim = self.feature_extractor.feature_channels
        
        # Project features to hidden dimension
        self.feature_projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        # Spatial encoding for relative poses
        self.spatial_encoder = SpatialEncoding(hidden_dim, hidden_dim)
        
        # Graph attention layers
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim*2, output_dim, kernel_size=1)
        
        # Store input_size for proper upsampling
        self.input_size = (224, 224)  # Default input size
        
        # Decoder heads
        self.predict_depth = predict_depth
        self.predict_segmentation = predict_segmentation
        
        # Depth decoder - updated to match input resolution
        if predict_depth:
            self.depth_decoder = nn.Sequential(
                nn.ConvTranspose2d(output_dim, 256, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)  # Final output layer
            )
        
        # Segmentation decoder - updated to match input resolution
        if predict_segmentation:
            self.seg_decoder = nn.Sequential(
                nn.ConvTranspose2d(output_dim, 256, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 2x upsampling
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)  # Final output layer (3 classes)
            )
            
        self.dropout = nn.Dropout(dropout)
    
    def extract_node_features(self, spatial_features):
        """Extract node features for the graph"""
        # Save input size for proper upsampling
        self.input_size = (spatial_features.size(2), spatial_features.size(3))
        
        # Extract CNN features - shape: [num_drones, channels, h, w]
        cnn_features = self.feature_extractor(spatial_features)
        
        # Project to hidden dimension
        projected_features = self.feature_projection(cnn_features)
        
        # Global average pooling for graph nodes
        node_features = F.adaptive_avg_pool2d(projected_features, (1, 1)).squeeze(-1).squeeze(-1)
        
        # Store the original feature maps for later skip connection
        self.original_feature_maps = projected_features
        
        return node_features
    
    def message_passing(self, node_features, edge_index, relative_poses=None):
        """
        Apply graph attention message passing with spatial encoding
        
        Args:
            node_features: Features for each drone [num_drones, hidden_dim]
            edge_index: Graph connectivity as COO format [2, num_edges]
            relative_poses: Relative poses between drones [num_edges, 6]
        """
        # If we have relative pose information, use it to enhance the message passing
        if relative_poses is not None and relative_poses.shape[0] > 0:
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            
            # Apply spatial encoding based on relative poses
            # Extract source node features
            source_features = node_features[source_nodes]
            
            # Apply spatial encoding (modulate features based on relative pose)
            modulated_features = self.spatial_encoder(source_features, relative_poses)
            
            # Update node features with spatially encoded versions
            node_features_encoded = node_features.clone()
            for i, (src, tgt) in enumerate(zip(source_nodes, target_nodes)):
                node_features_encoded[tgt] = modulated_features[i]
            
            # First GAT layer with encoded features
            x = F.elu(self.gat1(node_features_encoded, edge_index))
        else:
            # Standard GAT if no relative poses
            x = F.elu(self.gat1(node_features, edge_index))
        
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        
        return x
    
    def decode_features(self, node_features):
        """Decode node features back to original resolution"""
        batch_size = node_features.size(0)
        
        # Reshape node features to 2D spatial features
        gnn_features = node_features.view(batch_size, -1, 1, 1)
        
        # Expand to match original feature map dimensions
        h, w = self.original_feature_maps.size(2), self.original_feature_maps.size(3)
        gnn_features = gnn_features.expand(-1, -1, h, w)
        
        # Concatenate with original CNN features (skip connection)
        combined_features = torch.cat([self.original_feature_maps, gnn_features], dim=1)
        
        # Project combined features
        output_features = self.output_proj(combined_features)
        
        results = {'features': output_features}
        
        # Apply decoders if needed
        if self.predict_depth:
            depth = self.depth_decoder(output_features)
            
            # Ensure output matches input size exactly
            if depth.size(2) != self.input_size[0] or depth.size(3) != self.input_size[1]:
                depth = F.interpolate(depth, size=self.input_size, mode='bilinear', align_corners=True)
                
            results['depth'] = depth
            
        if self.predict_segmentation:
            segmentation = self.seg_decoder(output_features)
            
            # Ensure output matches input size exactly
            if segmentation.size(2) != self.input_size[0] or segmentation.size(3) != self.input_size[1]:
                segmentation = F.interpolate(segmentation, size=self.input_size, mode='bilinear', align_corners=True)
                
            results['segmentation'] = segmentation
            
        return results
    
    def forward(self, spatial_features, edge_index, relative_poses=None):
        """
        Forward pass through the entire model
        
        Args:
            spatial_features: Tensor of shape [num_drones, channels, height, width]
            edge_index: Graph connectivity as COO format [2, num_edges]
            relative_poses: Optional tensor of relative poses between drones [num_edges, 6]
        """
        # Extract node features
        node_features = self.extract_node_features(spatial_features)
        
        # Apply message passing
        updated_node_features = self.message_passing(node_features, edge_index, relative_poses)
        
        # Decode to output predictions
        output = self.decode_features(updated_node_features)
        
        return output