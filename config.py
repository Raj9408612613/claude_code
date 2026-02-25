"""
Environment Configuration
==========================
Centralised config for the MJX Spot navigation environment.
Imported by mjx_nav_env.py and dynamic_obstacles.py.
"""

# ── Humanoid walking obstacle ─────────────────────────────────────────────────
HUMANOID_OBSTACLE = {
    "enabled":        True,
    "speed":          0.8,    # m/s walking speed
    "stride_freq":    1.2,    # Hz  (used by single-env CPU class for animation)
    "patrol_radius":  1.5,    # metres from goal centre to each waypoint
    "mocap_z":        1.0,    # world-frame z of the torso mocap origin
    "wp_switch_dist": 0.2,    # switch waypoint when closer than this (m)
}
