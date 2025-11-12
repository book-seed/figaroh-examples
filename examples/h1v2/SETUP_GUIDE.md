# Setup Guide for ROBOT_TITLE Example

This guide will help you complete the setup and implementation of the ROBOT_TITLE example.

## ✅ What's Been Created

The following directory structure and files have been automatically generated:

### 📁 Directory Structure
- `config/` - Configuration files
- `data/` - Data storage (calibration, identification, results)
- `docs/` - Documentation
- `urdf/` - Robot URDF models
- `utils/` - Utility functions and tools
- `tmp/` - Temporary files

### 📄 Python Files
- `calibration.py` - Kinematic calibration script
- `identification.py` - Dynamic identification script  
- `optimal_config.py` - Optimal configuration generation
- `optimal_trajectory.py` - Optimal trajectory generation

### ⚙️ Configuration
- `config/robot_lower_config.yaml` - Main configuration file

### 📚 Documentation
- `README.md` - Main documentation
- `data/calibration/README.md` - Calibration data format guide
- `data/identification/README.md` - Identification data format guide

## 📝 Next Steps

Follow these steps to complete your implementation:

### Step 1: Robot Model Setup
1. Obtain the URDF file for your ROBOT_TITLE robot
2. Place it in the `urdf/` directory
3. Update `config/robot_lower_config.yaml`:
   - Set `model.urdf_path` to your URDF file
   - Update `model.package_name` if needed
   - List all active joints in `joints.active_joints`
   - Define joint limits in `joints.limits`

### Step 2: Test Robot Loading
Create a simple test script to verify your robot loads correctly:

```python
from figaroh.tools.robot import load_robot

robot = load_robot(
    robot_name="robot_lower",
    load_by_urdf=True,
    robot_pkg="robot_lower_description"
)
print(f"Loaded robot with {robot.nq} DOF")
```

### Step 3: Implement Utils
1. Edit `utils/robot_lower_tools.py`:
   - Update `load_robot_model()` with correct parameters
   - Implement `OptimalTrajectoryIPOPT` if needed
   
2. Edit `utils/simplified_colission_model.py`:
   - Implement simplified collision geometry if doing trajectory optimization

### Step 4: Collect Data
1. For calibration:
   - Collect end-effector position measurements
   - Save to `data/calibration/` in CSV format
   
2. For identification:
   - Record robot trajectories with joint positions, velocities, and torques
   - Save to `data/identification/` in CSV format

### Step 5: Implement Workflows
1. Complete `calibration.py`:
   - Load your robot model
   - Load measurement data
   - Configure calibration parameters
   - Run calibration and save results

2. Complete `identification.py`:
   - Load trajectory data
   - Build regressor matrix
   - Solve for dynamic parameters
   - Validate and save results

3. Complete `optimal_config.py`:
   - Define workspace constraints
   - Run optimization for observable configurations

4. Complete `optimal_trajectory.py`:
   - Define trajectory constraints
   - Run trajectory optimization

### Step 6: Test and Validate
1. Run each script individually
2. Verify outputs in `data/` directories
3. Validate results against ground truth or expected values

## 🔍 Reference Implementation

For detailed examples, refer to the H1v2 implementation:
- `examples/h1v2/calibration.py`
- `examples/h1v2/identification.py`
- `examples/h1v2/optimal_config.py`
- `examples/h1v2/optimal_trajectory.py`

## 📚 Resources

- FIGAROH documentation: [link]
- Pinocchio documentation: https://stack-of-tasks.github.io/pinocchio/
- Example data formats: See `data/*/README.md` files

## ❓ Common Questions

**Q: Where do I get the URDF file?**
A: Check your robot manufacturer's website, ROS packages, or create one from CAD models.

**Q: What if my robot has a mobile base?**
A: Include the mobile base joints in your URDF and active joints list.

**Q: How do I collect calibration data?**
A: Use external measurement systems (motion capture, laser tracker, etc.) or manual measurements.

**Q: What trajectory should I use for identification?**
A: Use the optimal trajectory generator or exciting trajectories with varying frequencies.

## 💡 Tips

1. Start simple - test each component individually
2. Use visualization to verify robot model and trajectories
3. Check data quality before running optimization
4. Validate results incrementally
5. Document your specific setup and parameters

## 🐛 Troubleshooting

If you encounter issues:
1. Check that all dependencies are installed
2. Verify URDF file loads correctly
3. Ensure data files are in correct format
4. Review configuration file parameters
5. Refer to H1v2 example for working implementation

## ✨ Contributing Back

If you successfully implement this for your robot:
1. Consider sharing your configuration
2. Document robot-specific tips
3. Contribute back to the examples repository

Good luck with your ROBOT_TITLE calibration and identification! 🚀
