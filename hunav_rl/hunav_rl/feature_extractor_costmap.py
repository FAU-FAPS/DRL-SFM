"""Feature extraction for costmap-based RL observation."""


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym

  

class LeNetCostmapExtractor(nn.Module):
    """LeNet-based convolutional neural network for costmap feature
    extraction."""
    
    def __init__(self):
        """Initialize the LeNet costmap extractor."""
        super(LeNetCostmapExtractor, self).__init__()
        # Input: (N, 1, 120, 120)
        self.conv_net = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            ),  # -> (N,16,120,120)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N,16,60,60)
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            ),  # -> (N,32,60,60)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N,32,30,30)
            nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            ),  # -> (N,64,30,30)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N,64,15,15)
            nn.Flatten()  # -> (N,64*15*15=14400)
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 15 * 15, 128),
            nn.ReLU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, 1, 120, 120).
            
        Returns:
            Feature tensor of shape (N, 128).
        """
        features = self.conv_net(x)
        return self.linear(features)
    
  

class CustomCombinedExtractorCostmap(BaseFeaturesExtractor):
    """Combined feature extractor for costmap and waypoint information."""
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        """Initialize the combined extractor.
        
        Args:
            observation_space: Gymnasium dictionary observation space.
            features_dim: Dimension of output features.
        """
        super(CustomCombinedExtractorCostmap, self).__init__(
            observation_space, features_dim
        )
        self.agent_extractor = nn.Sequential(
            nn.Flatten()
        )
        self.costmap_extractor = LeNetCostmapExtractor()
        agent_dim = (
            observation_space.spaces["waypoint_distances"].shape[0]
            + observation_space.spaces["waypoint_directions"].shape[0]
        )
        costmap_dim = 128
        combined_input_dim = agent_dim + costmap_dim
        
        self.combined_linear = nn.Sequential(
            nn.Linear(combined_input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Forward pass combining agent and costmap features.
        
        Args:
            observations: Dictionary containing 'waypoint_distances',
            'waypoint_directions', and 'costmap'.
            
        Returns:
            Combined feature tensor.
        """
        # Process waypoint features: concatenate radius and angle
        waypoint_distances = observations["waypoint_distances"]
        waypoint_directions = observations["waypoint_directions"]
        agent_features = self.agent_extractor(
            th.cat([waypoint_distances, waypoint_directions], dim=1)
        )

        # Process costmap
        costmap_obs = observations["costmap"]
        if costmap_obs.ndim == 4 and costmap_obs.shape[-1] == 1:
            costmap_obs = costmap_obs.permute(0, 3, 1, 2)
        costmap_features = self.costmap_extractor(costmap_obs)

        # Combine both modalities
        combined = th.cat([agent_features, costmap_features], dim=1)
        return self.combined_linear(combined)
