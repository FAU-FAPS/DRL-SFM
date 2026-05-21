"""Feature extraction for humap-based RL observation."""


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym

  

class LeNetHumapExtractor(nn.Module):
    """LeNet-based convolutional neural network for human map feature
    extraction."""
    
    def __init__(self):
        """Initialize the LeNet humap extractor."""
        super(LeNetHumapExtractor, self).__init__()
        # Input: (N, 3, 120, 120)
        self.conv_net = nn.Sequential(
            nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            # -> (N, 16, 120, 120)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N, 16, 60, 60)
            nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),  # -> (N, 32, 60, 60)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N, 32, 30, 30)
            nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # -> (N, 64, 30, 30)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N, 64, 15, 15)
            nn.Flatten()  # -> (N, 64*15*15 = 14400)
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 15 * 15, 128),
            nn.ReLU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, 3, 120, 120).
            
        Returns:
            Feature tensor of shape (N, 128).
        """
        features = self.conv_net(x)
        return self.linear(features)

  

class CustomCombinedExtractorHumap(BaseFeaturesExtractor):
    """Combined feature extractor for human map and waypoint information."""
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256
    ):
        """Initialize the combined extractor.
        
        Args:
            observation_space: Gymnasium dictionary observation space.
            features_dim: Dimension of output features.
        """
        super(CustomCombinedExtractorHumap, self).__init__(
            observation_space, features_dim
        )
        # Flatten waypoint_distances and waypoint_directions
        self.agent_extractor = nn.Sequential(
            nn.Flatten()
        )
        # LeNet-based humap extractor for 3×120×120 input
        self.humap_extractor = LeNetHumapExtractor()
        
        # Compute dims
        agent_dim = (
            observation_space.spaces["waypoint_distances"].shape[0]
            + observation_space.spaces["waypoint_directions"].shape[0]
        )
        humap_dim = 128  # output of LeNetHumapExtractor
        combined_input_dim = agent_dim + humap_dim
        
        self.combined_linear = nn.Sequential(
            nn.Linear(combined_input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Forward pass combining agent and humap features.
        
        Args:
            observations: Dictionary containing 'waypoint_distances',
            'waypoint_directions', and 'humap'.
            
        Returns:
            Combined feature tensor.
        """
        # Process agent features: concatenate radius and angle
        waypoint_distances = observations["waypoint_distances"]
        waypoint_directions = observations["waypoint_directions"]
        agent_features = self.agent_extractor(
            th.cat([waypoint_distances, waypoint_directions], dim=1)
        )

        # Process humap: expect (N, 120, 120, 3) or (N, 3, 120, 120)
        humap_obs = observations["humap"]
        if humap_obs.ndim == 4 and humap_obs.shape[-1] == 3:
            humap_obs = humap_obs.permute(0, 3, 1, 2)
        humap_features = self.humap_extractor(humap_obs)

        # Combine both modalities
        combined = th.cat([agent_features, humap_features], dim=1)
        return self.combined_linear(combined)

