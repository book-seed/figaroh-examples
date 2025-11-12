# ROBOT_TITLE Robot Calibration and Identification Framework

This directory provides a comprehensive framework for calibrating and identifying the kinematic and dynamic parameters of the ROBOT_TITLE robot using the FIGAROH library.

## Overview

The ROBOT_TITLE robot requires accurate modeling for precision tasks. This framework provides four interconnected tools:

1. **Kinematic Calibration** - Correct geometric errors in robot structure
2. **Dynamic Parameter Identification** - Identify physical properties for accurate dynamics  
3. **Optimal Configuration Generation** - Scientifically select measurement poses
4. **Optimal Trajectory Generation** - Create information-rich motion for identification

## Directory Structure

```
robot_lower/
├── calibration.py              # Kinematic calibration implementation
├── identification.py           # Dynamic parameter identification
├── optimal_config.py           # Optimal configuration generation
├── optimal_trajectory.py       # Optimal trajectory generation
├── README.md                   # This file
├── config/                     # Configuration files
│   ├── robot_lower_config.yaml
│   └── templates/
├── data/                       # Data storage
│   ├── calibration/
│   ├── identification/
│   └── optimal_configurations/
├── docs/                       # Documentation
├── urdf/                       # Robot URDF files
└── utils/                      # Utility functions
    ├── __init__.py
    ├── robot_lower_tools.py
    └── simplified_colission_model.py
```

## Quick Start

### Prerequisites

```bash
# Install FIGAROH and dependencies
pip install figaroh numpy scipy matplotlib pandas

# Install Pinocchio and CyIpopt (recommended via conda)
conda install -c conda-forge pinocchio cyipopt
```

### Basic Usage

1. **Kinematic Calibration**:
```bash
python calibration.py
```

2. **Dynamic Identification**:
```bash
python identification.py
```

3. **Generate Optimal Configurations**:
```bash
python optimal_config.py
```

4. **Generate Optimal Trajectories**:
```bash
python optimal_trajectory.py
```

## Implementation Steps

This is a template directory. To complete the implementation for your ROBOT_TITLE robot:

### 1. Robot Model Setup
- [ ] Add URDF file to `urdf/` directory
- [ ] Update robot package name in config files
- [ ] Define active joints in `config/robot_lower_config.yaml`
- [ ] Set joint limits and constraints

### 2. Calibration Implementation
- [ ] Collect measurement data (camera, laser tracker, etc.)
- [ ] Define calibration parameters in config
- [ ] Implement calibration workflow in `calibration.py`
- [ ] Validate calibration results

### 3. Identification Implementation  
- [ ] Record robot motion data (positions, velocities, torques)
- [ ] Prepare data files in `data/identification/`
- [ ] Implement identification workflow in `identification.py`
- [ ] Validate identified parameters

### 4. Optimal Configuration
- [ ] Define workspace constraints
- [ ] Implement observability optimization in `optimal_config.py`
- [ ] Generate and save optimal configurations

### 5. Optimal Trajectory
- [ ] Build simplified collision model (if needed)
- [ ] Implement trajectory optimization in `optimal_trajectory.py`
- [ ] Validate trajectory on real robot

## Configuration

Edit `config/robot_lower_config.yaml` to customize:
- Robot model parameters
- Active joints and limits
- Calibration parameters
- Identification settings
- Trajectory constraints
- Data paths

## Utility Functions

The `utils/` directory contains robot-specific helpers:
- `robot_lower_tools.py`: Custom tools and optimization classes
- `simplified_colission_model.py`: Collision model for trajectory planning

## Data Format

### Calibration Data
Store measurement data in `data/calibration/`:
- CSV format with columns: [timestamp, x, y, z, ...]
- Include joint angles and measured positions

### Identification Data  
Store trajectory data in `data/identification/`:
- CSV format with columns: [timestamp, q1, q2, ..., dq1, dq2, ..., tau1, tau2, ...]
- Positions, velocities, and torques for all joints

## Reference

This template is based on the H1v2 example. For detailed implementation examples, see:
- `examples/h1v2/` - Complete H1v2 implementation
- FIGAROH documentation: [link to docs]

## Troubleshooting

### Common Issues

1. **Robot model not loading**
   - Check URDF file path in config
   - Verify robot package is installed
   - Check for mesh file dependencies

2. **Optimization fails**
   - Verify joint limits are correct
   - Check initial configuration is valid
   - Reduce trajectory complexity

3. **Import errors**
   - Ensure FIGAROH is installed
   - Check Python path includes parent directory
   - Verify all dependencies are installed

## Contributing

To improve this example:
1. Implement the TODO items in each script
2. Add your specific robot configuration
3. Document your workflow and results
4. Share successful calibration/identification results

## License

[Add your license information here]

## Contact

[Add contact information or links to documentation]
