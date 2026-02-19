"""
Spot Isaac Sim Training Package
================================
Train Boston Dynamics Spot robot for autonomous navigation
using 5 depth cameras in NVIDIA Isaac Sim with full domain randomization.

Camera layout:
    Front (0°), Front-Left (70°), Front-Right (-70°),
    Rear-Left (145°), Rear-Right (-145°)
    → Near 360° depth coverage

Modules:
    config          - All simulation, training, and sensor configuration
    depth_cameras   - 5-camera rig with realistic noise models
    environment     - Room loading, furniture placement, domain randomization
    dynamic_obstacles - Moving obstacle system (people, carts)
    reward          - Navigation reward computation
    navigation_env  - Main Gymnasium-compatible Isaac Sim environment
    train           - PPO training entry point
"""
