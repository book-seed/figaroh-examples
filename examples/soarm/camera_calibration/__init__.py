#!/usr/bin/env python3
"""
FIGAROH Camera Calibration Package

This package contains tools for camera calibration and ArUco marker detection
for the FIGAROH robotics project.

Tools included:
- camera_calibration_tool: Full camera calibration using chessboard patterns
- webcam_parameter_estimator: Quick parameter estimation for common webcams
- generate_aruco_markers: ArUco marker generation for testing
- detection_monitor: Real-time marker detection monitoring

Usage:
    # Import individual tools
    from camera_calibration.tools import camera_calibration_tool
    from camera_calibration.tools import webcam_parameter_estimator
    
    # Or run tools directly from command line
    python -m camera_calibration.tools.camera_calibration_tool
"""

__version__ = "1.0.0"
__author__ = "FIGAROH Project"
__description__ = "Camera calibration tools for FIGAROH robotics"

# Make tools easily accessible
import sys
import os

# Add tools directory to path for imports
_tools_dir = os.path.join(os.path.dirname(__file__), 'tools')
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

# Export main components
__all__ = [
    'camera_calibration_tool',
    'webcam_parameter_estimator', 
    'generate_aruco_markers',
    'detection_monitor'
]
