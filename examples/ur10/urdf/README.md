# UR10 URDF Files

## Kept as copies (cannot symlink)

| File | models/ counterpart | Reason |
|---|---|---|
| `ur10_robot.urdf` | `models/ur_description/urdf/ur10_robot.urdf` | Completely different content. The examples version is an **UR10e** model with a D435 camera, mount, tool, and support structure (Gazebo plugins, camera sensors). The models/ version is a plain **UR10** with only the base arm kinematic chain and transmissions. |

These models serve different purposes — the examples file includes the full sensor suite
used by the figaroh calibration/identification pipeline.
