# Camera Calibration Tools for FIGAROH

This directory contains all camera calibration related tools and data for the FIGAROH robotics project.

## Directory Structure

```
camera_calibration/
├── tools/           # Calibration and marker generation tools
├── patterns/        # Calibration patterns (chessboard, ArUco markers)
├── data/           # Saved calibration data and results
├── tests/          # Test scripts for camera functionality
└── README.md       # This file
```

## Tools Overview

### 🔧 Main Tools (`tools/`)

#### `camera_calibration_tool.py`
The main camera calibration tool that performs complete camera calibration using chessboard patterns.

**Features:**
- Real-time camera capture with preview
- Chessboard detection and corner finding
- Camera matrix and distortion coefficient calculation
- Multiple export formats (JSON, CSV, Python, Pickle)
- Quality assessment and calibration verification

**Usage:**
```bash
cd camera_calibration/tools
python camera_calibration_tool.py --help
python camera_calibration_tool.py --camera 0 --pattern-size 9,6
```

#### `webcam_parameter_estimator.py`
Estimates camera parameters for common webcam specifications when exact calibration isn't available.

**Features:**
- Automatic resolution detection
- Preset parameters for common webcam types
- Field-of-view estimation tools
- Parameter export for FIGAROH integration

**Usage:**
```bash
cd camera_calibration/tools
python webcam_parameter_estimator.py --camera 0
python webcam_parameter_estimator.py --preset basic_720p
```

#### `generate_aruco_markers.py`
Generates ArUco markers for calibration and testing.

**Features:**
- Single marker generation with customizable ID and size
- Marker set generation with combined layout
- Print-ready format with sizing recommendations
- Multiple ArUco dictionary support

**Usage:**
```bash
cd camera_calibration/tools
python generate_aruco_markers.py
```

#### `detection_monitor.py`
Real-time ArUco marker detection monitor for testing and debugging.

**Features:**
- Live camera feed with marker detection overlay
- Detection status and timing information
- Multiple marker ID tracking
- Detection quality assessment

**Usage:**
```bash
cd camera_calibration/tools
python detection_monitor.py
```

### 🧪 Tests (`tests/`)

#### `test_camera_window.py`
Simple camera functionality test to verify OpenCV and camera setup.

**Usage:**
```bash
cd camera_calibration/tests
python test_camera_window.py
```

### 📐 Patterns (`patterns/`)

- `chessboard_pattern.png` - Standard chessboard pattern for camera calibration
- `aruco_marker_0.png` - ArUco marker with ID 0 for testing
- Additional generated markers from `generate_aruco_markers.py`

### 💾 Data (`data/`)

Contains saved calibration results in various formats:
- `.json` - Human-readable parameter files
- `.csv` - Spreadsheet-compatible data
- `.pkl` - Python pickle files for direct loading
- `.py` - Python code files for integration

## Quick Start Guide

### 1. Camera Setup Test
```bash
cd camera_calibration/tests
python test_camera_window.py
```

### 2. Generate Calibration Patterns
```bash
cd camera_calibration/tools
python generate_aruco_markers.py
# Print the generated markers or display on screen
```

### 3. Full Camera Calibration
```bash
cd camera_calibration/tools
python camera_calibration_tool.py --camera 0 --pattern-size 9,6
# Follow the on-screen instructions to capture calibration images
```

### 4. Quick Parameter Estimation (Alternative)
```bash
cd camera_calibration/tools
python webcam_parameter_estimator.py --camera 0
```

### 5. Test ArUco Detection
```bash
cd camera_calibration/tools
python detection_monitor.py
# Hold ArUco marker ID 0 in front of camera
```

## Integration with FIGAROH

The calibration tools are designed to integrate seamlessly with the main FIGAROH system:

1. **Parameter Export**: Calibration results can be exported in Python format for direct use
2. **Data Storage**: All calibration data is saved in the `data/` directory
3. **Configuration**: Parameters can be automatically updated in `camera_data_collection.py`

## Requirements

- Python 3.8+
- OpenCV 4.5+ (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)

## Tips for Best Results

1. **Lighting**: Ensure good, even lighting when calibrating
2. **Pattern Size**: Use appropriate chessboard size for your camera resolution
3. **Distance Variation**: Capture images at various distances during calibration
4. **Angle Variation**: Tilt and rotate the calibration pattern
5. **Stability**: Keep the camera steady during capture

## Troubleshooting

- **Camera not detected**: Check camera permissions and connections
- **Pattern not detected**: Improve lighting or print quality
- **Poor calibration**: Capture more images with better variation
- **Import errors**: Ensure all dependencies are installed

For more help, see the individual tool documentation or run any tool with `--help`.
