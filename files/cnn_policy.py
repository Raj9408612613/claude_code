"""
Custom CNN + MLP Feature Extractor for Spot Obstacle Avoidance

Processes a Dict observation space:
    "image"          -> CNN  -> visual features (128-dim)
    "proprioception" -> MLP  -> body-state features (64-dim)
                               ──────────────────────
                               concatenated -> 192-dim shared features

The combined feature vector feeds into PPO's actor and critic heads.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpotCNNExtractor(BaseFeaturesExtractor):
    """
    Combined CNN (for camera images) + MLP (for proprioception) extractor.

    Architecture:
        Image path  (H x W x 3):
            Conv2d(3, 32, 8, stride=4)  -> ReLU
            Conv2d(32, 64, 4, stride=2) -> ReLU
            Conv2d(64, 64, 3, stride=1) -> ReLU
            Flatten -> Linear(*, 128)   -> ReLU

        Proprioception path (37,):
            Linear(37, 64) -> ReLU
            Linear(64, 64) -> ReLU

        Concatenated: 128 + 64 = 192 features
    """

    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 128,
                 proprio_hidden_dim: int = 64):
        # Calculate total features dimension before calling super().__init__
        features_dim = cnn_output_dim + proprio_hidden_dim
        super().__init__(observation_space, features_dim=features_dim)

        image_space = observation_space["image"]
        proprio_space = observation_space["proprioception"]

        # SB3's VecTransposeImage converts (H, W, C) -> (C, H, W) before
        # the observation reaches the policy, so handle both formats.
        if image_space.shape[0] in (1, 3):  # channels-first (C, H, W)
            n_channels = image_space.shape[0]
            image_h = image_space.shape[1]
            image_w = image_space.shape[2]
        else:  # channels-last (H, W, C)
            n_channels = image_space.shape[2]
            image_h = image_space.shape[0]
            image_w = image_space.shape[1]

        # --- CNN for camera images ---
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size by doing a forward pass with dummy data
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, image_h, image_w)
            cnn_flat_size = self.cnn(dummy).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_flat_size, cnn_output_dim),
            nn.ReLU(),
        )

        # --- MLP for proprioception ---
        proprio_dim = proprio_space.shape[0]
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, proprio_hidden_dim),
            nn.ReLU(),
            nn.Linear(proprio_hidden_dim, proprio_hidden_dim),
            nn.ReLU(),
        )

        # Orthogonal init for faster convergence on GPU
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"]

        # Ensure float (SB3 normally handles this, but be safe)
        if img.dtype != torch.float32:
            img = img.float()

        # If image is in (B, H, W, C) format, permute to (B, C, H, W)
        if img.dim() == 4 and img.shape[-1] in (1, 3):
            img = img.permute(0, 3, 1, 2)

        # Normalise to [0, 1] if pixel values are in [0, 255]
        if img.max() > 1.0:
            img = img * (1.0 / 255.0)

        cnn_features = self.cnn_linear(self.cnn(img))

        # Proprioception
        proprio = observations["proprioception"]
        if proprio.dtype != torch.float32:
            proprio = proprio.float()
        proprio_features = self.proprio_mlp(proprio)

        return torch.cat([cnn_features, proprio_features], dim=1)
