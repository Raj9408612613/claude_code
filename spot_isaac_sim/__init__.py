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

Usage:
    # SimulationApp must be started before omni.* imports work.
    # Call ensure_isaac_sim() before importing any sim modules.
    from spot_isaac_sim import ensure_isaac_sim
    ensure_isaac_sim(headless=True)
    from spot_isaac_sim.navigation_env import SpotNavigationEnv
"""

_simulation_app = None


def ensure_isaac_sim(headless=True, **kwargs):
    """Start SimulationApp if not already running. Must be called before importing omni.* modules."""
    global _simulation_app
    if _simulation_app is not None:
        return _simulation_app
    try:
        from isaacsim import SimulationApp
        launch_config = {"headless": headless}
        launch_config.update(kwargs)
        _simulation_app = SimulationApp(launch_config)
        print(f"Isaac Sim started (headless={headless})")
        return _simulation_app
    except ImportError:
        print("Isaac Sim not installed — running in fallback mode")
        return None
