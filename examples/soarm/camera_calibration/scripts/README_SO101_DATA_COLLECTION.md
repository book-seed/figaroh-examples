# SO101 Robot Camera Data Collection

This directory contains scripts for collecting calibration data using the SO101 Follower robot and camera with ArUco marker detection.

## Files Created

### 1. `so101_camera_data_collection.py`
Complete SO101 Follower robot interface that uses the full robot configuration and functionality from lerobot.

**Features:**
- Uses the full `SO101Follower` class with proper configuration
- Includes robot calibration and safety features
- Connects to cameras as configured in the robot config
- Reads motor positions using `robot.bus.sync_read("Present_Position")`

### 2. `motor_bus_camera_data_collection.py`
Direct motor bus interface that communicates directly with the Feetech motors without requiring full robot setup.

**Features:**
- Uses `FeetechMotorsBus` directly for faster setup
- Configures motors with proper IDs and models (STS3215)
- Normalized position readings (-100 to +100 range)
- Simplified motor communication

### 3. `camera_data_collection.py` (Original)
Mock robot interface for testing camera and ArUco detection without real hardware.

## Key Functions Used from SO101 Follower and Motors Bus

### From `so101_follower.py`:
```python
# Get robot observation (includes motor positions)
obs_dict = robot.bus.sync_read("Present_Position")

# Connect to robot
robot.connect(calibrate=False)

# Disconnect robot
robot.disconnect()
```

### From `motors_bus.py`:
```python
# Read all motor positions synchronously
positions = bus.sync_read("Present_Position")

# Read individual motor position
position = bus.read("Present_Position", "motor_name", normalize=True)

# Connect to motor bus
bus.connect()

# Disconnect motor bus
bus.disconnect()
```

## Usage

### For Real SO101 Robot:
```bash
python so101_camera_data_collection.py
```

### For Direct Motor Bus Access:
```bash
python motor_bus_camera_data_collection.py
```

### For Testing Without Hardware:
```bash
python camera_data_collection.py
```

## Data Collection Process

1. **Connect to Robot/Motors**: Scripts will attempt to connect to the specified serial port
2. **Initialize Camera**: OpenCV camera capture on device 0
3. **ArUco Detection**: Detects ArUco markers (specifically ID 0)
4. **Data Collection**: Press SPACE when marker is visible to collect data points
5. **Save Data**: Press S to save collected data to JSON and CSV files

## Data Format

Each collected data point includes:
- `timestamp`: Unix timestamp
- `marker_position`: 3D position [x, y, z] in meters
- `marker_rotation`: 3D rotation vector [rx, ry, rz] in radians
- `joint_angles`: List of 6 joint positions (normalized or in radians)
- `ee_position`: End effector position (estimated)
- `ee_orientation`: End effector orientation (estimated)
- `robot_info`/`motor_info`: Robot/motor status information

## Output Files

Data is saved with timestamps:
- `so101_calibration_data_YYYYMMDD_HHMMSS.json`
- `so101_calibration_data_YYYYMMDD_HHMMSS.csv`
- `motor_bus_calibration_data_YYYYMMDD_HHMMSS.json`
- `motor_bus_calibration_data_YYYYMMDD_HHMMSS.csv`

## Configuration

### Robot Port
Default: `/dev/tty.usbmodem575E0031751`
You'll be prompted to enter the correct port when running the scripts.

### Camera Calibration
Update camera matrix and distortion coefficients in the scripts:
```python
camera_matrix = np.array([[800, 0, 320],
                         [0, 800, 240],
                         [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
```

### ArUco Marker
- Default marker size: 5cm (0.05m)
- Dictionary: DICT_6X6_250
- Target ID: 0

## Requirements

- OpenCV with ArUco support
- NumPy
- Pandas
- lerobot package
- Connected SO101 robot or compatible Feetech motors
- Camera (USB or built-in)
- Printed ArUco marker (ID 0)

## Troubleshooting

1. **Robot Connection Issues**: Check serial port and ensure robot is powered
2. **Camera Issues**: Verify camera index (try different values for `cv2.VideoCapture()`)
3. **ArUco Detection**: Ensure good lighting and marker is flat/visible
4. **Motor Errors**: Check motor IDs and baudrate settings

## Integration with FIGAROH

The collected data can be used with FIGAROH's calibration tools for:
- Hand-eye calibration
- Robot kinematics identification
- Camera-robot coordination
- Pose estimation validation
