#!/usr/bin/env python3
"""
Real-time data collection with camera view and ArUco marker detection using SO101 Follower robot
"""

import cv2
import numpy as np
import pandas as pd
import json
import time
import logging
import sys
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Add the lerobot path to access the robot modules
sys.path.append('/Users/thanhndv212/Develop/lerobot')

from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig


class ArucoTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_length):
        """
        Initialize ArUco tracker
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients
            marker_length: physical size of ArUco marker in meters
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        
        # Create ArUco dictionary and detector parameters (OpenCV 4.x compatible)
        try:
            # OpenCV 4.7+
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            
            # Improve detection parameters
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.adaptiveThreshConstant = 7
            self.aruco_params.minMarkerPerimeterRate = 0.03
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.03
            self.aruco_params.minCornerDistanceRate = 0.05
            self.aruco_params.minDistanceToBorder = 3
            self.aruco_params.minMarkerDistanceRate = 0.05
            
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
            print("✅ Using OpenCV 4.7+ ArUco API with enhanced detection parameters")
        except AttributeError:
            # OpenCV 4.6 and earlier
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            
            # Improve detection parameters for older API
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.adaptiveThreshConstant = 7
            self.aruco_params.minMarkerPerimeterRate = 0.03
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.03
            self.aruco_params.minCornerDistanceRate = 0.05
            self.aruco_params.minDistanceToBorder = 3
            self.aruco_params.minMarkerDistanceRate = 0.05
            
            self.use_new_api = False
            print("✅ Using OpenCV 4.6 ArUco API with enhanced detection parameters")
    
    def detect_and_estimate_pose(self, frame):
        """
        Detect ArUco markers and estimate their 6D pose
        
        Returns:
            poses: list of (rvec, tvec) tuples for each detected marker
            ids: list of marker IDs
            corners: detected marker corners
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect markers (OpenCV version compatible)
        if self.use_new_api:
            # OpenCV 4.7+
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            # OpenCV 4.6 and earlier
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        # Debug: print detection info
        if ids is not None:
            detected_ids = ids.flatten().tolist()
            print(f"🔍 Detected ArUco markers: {detected_ids}")
        else:
            print("🔍 No ArUco markers detected")
        
        poses = []
        if ids is not None:
            # Estimate pose for each marker (OpenCV version compatible)
            try:
                # Try using estimatePoseSingleMarkers first (most reliable)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                )
                for i in range(len(ids)):
                    poses.append((rvecs[i][0], tvecs[i][0]))
                        
            except AttributeError:
                # Fallback to manual solvePnP if estimatePoseSingleMarkers not available
                for i in range(len(corners)):
                    half_size = self.marker_length / 2
                    object_points = np.array([
                        [-half_size, half_size, 0],
                        [half_size, half_size, 0], 
                        [half_size, -half_size, 0],
                        [-half_size, -half_size, 0]
                    ], dtype=np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, corners[i], 
                        self.camera_matrix, self.dist_coeffs
                    )
                    
                    if success:
                        poses.append((rvec.flatten(), tvec.flatten()))
                    else:
                        poses.append((np.zeros(3), np.zeros(3)))
        
        return poses, ids, corners
    
    def draw_pose(self, frame, rvec, tvec):
        """Draw coordinate axes on the frame"""
        try:
            # Try OpenCV 4.7+ method
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                            rvec, tvec, self.marker_length * 0.5)
        except AttributeError:
            try:
                # Try older OpenCV method
                cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, 
                                 rvec, tvec, self.marker_length * 0.5)
            except AttributeError:
                # Manual drawing of axes if no drawAxis functions are available
                axis_length = self.marker_length * 0.5
                
                # Define 3D points for axes
                axis_points = np.array([
                    [0, 0, 0],           # Origin
                    [axis_length, 0, 0], # X axis
                    [0, axis_length, 0], # Y axis  
                    [0, 0, -axis_length] # Z axis (negative for right-hand rule)
                ], dtype=np.float32)
                
                # Project 3D points to 2D
                projected_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                )
                
                # Convert to integer coordinates
                points = projected_points.reshape(-1, 2).astype(int)
                
                # Draw axes
                if len(points) >= 4:
                    origin = tuple(points[0])
                    x_end = tuple(points[1])
                    y_end = tuple(points[2])
                    z_end = tuple(points[3])
                    
                    # Draw X axis (red)
                    cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 3)
                    # Draw Y axis (green)
                    cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 3)
                    # Draw Z axis (blue)
                    cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 3)
                    
                    # Add axis labels
                    cv2.putText(frame, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame


class SO101RobotInterface:
    """Real SO101 Follower robot interface for data collection"""
    
    def __init__(self, port: str = "/dev/tty.usbmodem575E0031751"):
        """
        Initialize the SO101 Follower robot interface
        
        Args:
            port: Serial port for the robot communication
        """
        # Create robot configuration
        self.config = SO101FollowerConfig(port=port)
        
        # Initialize the robot
        self.robot = SO101Follower(self.config)
        self.is_connected = False
        
        # Motor names in the order they appear
        self.motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        
        print(f"🤖 SO101 Robot Interface initialized with port: {port}")
    
    def connect(self) -> bool:
        """Connect to the robot"""
        try:
            print("🔌 Connecting to SO101 Follower robot...")
            self.robot.connect(calibrate=False)  # Skip calibration for data collection
            self.is_connected = True
            print("✅ Robot connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to robot: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the robot"""
        if self.is_connected:
            try:
                self.robot.disconnect()
                self.is_connected = False
                print("✅ Robot disconnected successfully!")
            except Exception as e:
                print(f"❌ Error disconnecting robot: {e}")
    
    def get_joint_angles(self) -> List[float]:
        """
        Get current joint angles from the robot
        
        Returns:
            List of joint angles in radians for all motors
        """
        if not self.is_connected:
            print("❌ Robot not connected! Cannot read joint angles.")
            return [0.0] * 6  # Return zeros if not connected
        
        try:
            # Read motor positions using the robot's bus
            positions = self.robot.bus.sync_read("Present_Position")
            
            # Convert to list in the correct order
            joint_angles = []
            for motor_name in self.motor_names:
                if motor_name in positions:
                    joint_angles.append(float(positions[motor_name]))
                else:
                    joint_angles.append(0.0)
            
            return joint_angles
            
        except Exception as e:
            print(f"❌ Error reading joint angles: {e}")
            return [0.0] * 6  # Return zeros on error
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end effector pose (forward kinematics would be needed for real implementation)
        For now, return estimated pose based on joint angles
        
        Returns:
            position: 3D position (x, y, z)
            orientation: 3D orientation (rx, ry, rz)
        """
        # This is a simplified implementation
        # In a real system, you would use forward kinematics
        joint_angles = self.get_joint_angles()
        
        if len(joint_angles) >= 3:
            # Simple approximation based on first 3 joints
            # This is not accurate and should be replaced with proper forward kinematics
            x = 0.3 + 0.2 * np.sin(joint_angles[0]) * np.cos(joint_angles[1])
            y = 0.2 * np.sin(joint_angles[0]) * np.sin(joint_angles[1])
            z = 0.4 + 0.2 * np.cos(joint_angles[1])
            
            position = np.array([x, y, z])
            orientation = np.array([joint_angles[3] if len(joint_angles) > 3 else 0.0,
                                  joint_angles[4] if len(joint_angles) > 4 else 0.0,
                                  joint_angles[0]])  # Base rotation
        else:
            position = np.array([0.3, 0.0, 0.4])
            orientation = np.array([0.0, 0.0, 0.0])
        
        return position, orientation
    
    def get_robot_info(self) -> Dict[str, Any]:
        """Get information about the robot state"""
        info = {
            "connected": self.is_connected,
            "motor_count": len(self.motor_names),
            "motor_names": self.motor_names,
            "bus_connected": self.robot.bus.is_connected if hasattr(self.robot, 'bus') else False
        }
        
        if self.is_connected:
            try:
                joint_angles = self.get_joint_angles()
                info["current_joints"] = joint_angles
                ee_pos, ee_ori = self.get_end_effector_pose()
                info["end_effector_position"] = ee_pos.tolist()
                info["end_effector_orientation"] = ee_ori.tolist()
            except Exception as e:
                info["error"] = str(e)
        
        return info


class CalibrationDataCollector:
    def __init__(self, aruco_tracker: ArucoTracker, robot_interface: SO101RobotInterface, target_marker_id: int = 0):
        """
        Initialize calibration data collector
        
        Args:
            aruco_tracker: ArUco marker tracker instance
            robot_interface: SO101 Robot communication interface
            target_marker_id: ID of the ArUco marker to track (can be changed dynamically)
        """
        self.aruco_tracker = aruco_tracker
        self.robot_interface = robot_interface
        self.target_marker_id = target_marker_id
        
        self.collected_data = []
        self.is_collecting = False
        self.detected_markers = set()  # Keep track of all detected markers
        
    def update_target_marker(self, new_id: int):
        """Update the target marker ID"""
        self.target_marker_id = new_id
        print(f"🎯 Target marker updated to ID: {new_id}")
    
    def collect_data_point(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Collect a single data point (marker pose + joint angles)
        
        Returns:
            Dictionary containing the collected data or None if marker not detected
        """
        # Get ArUco marker pose
        poses, ids, corners = self.aruco_tracker.detect_and_estimate_pose(frame)
        
        # Update detected markers set
        if ids is not None:
            self.detected_markers.update(ids.flatten().tolist())
        
        if ids is None or self.target_marker_id not in ids.flatten():
            return None
        
        # Find the target marker
        marker_idx = np.where(ids.flatten() == self.target_marker_id)[0][0]
        rvec, tvec = poses[marker_idx]
        
        # Get joint angles from robot
        joint_angles = self.robot_interface.get_joint_angles()
        
        # Get end effector pose (if available)
        try:
            ee_position, ee_orientation = self.robot_interface.get_end_effector_pose()
        except Exception:
            ee_position, ee_orientation = None, None
        
        # Create data point
        data_point = {
            'timestamp': time.time(),
            'marker_position': tvec.tolist(),
            'marker_rotation': rvec.tolist(),
            'joint_angles': joint_angles,
            'ee_position': ee_position.tolist() if ee_position is not None else None,
            'ee_orientation': ee_orientation.tolist() if ee_orientation is not None else None,
            'marker_id': int(self.target_marker_id),
            'robot_info': self.robot_interface.get_robot_info(),
            'all_detected_markers': list(self.detected_markers)
        }
        
        return data_point

    def add_data_point(self, data_point: Dict[str, Any]):
        """Add a data point to the collection"""
        self.collected_data.append(data_point)
        print(f"✅ Collected data point {len(self.collected_data)}")
        
        # Print some info about the data point
        pos = data_point['marker_position']
        joints = data_point['joint_angles'][:3]  # Show first 3 joints
        print(f"   Marker pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
        print(f"   Joints 1-3: [{joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}] rad")
        
        # Show robot connection status
        robot_info = data_point.get('robot_info', {})
        print(f"   Robot connected: {robot_info.get('connected', 'unknown')}")
        print(f"   Bus connected: {robot_info.get('bus_connected', 'unknown')}")
    
    def save_data(self, filename: str):
        """Save collected data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = f"{filename}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        
        # Save as CSV for easier analysis
        csv_filename = f"{filename}_{timestamp}.csv"
        df_data = []
        
        for point in self.collected_data:
            row = {
                'timestamp': point['timestamp'],
                'marker_id': point['marker_id'],
                'marker_pos_x': point['marker_position'][0],
                'marker_pos_y': point['marker_position'][1],
                'marker_pos_z': point['marker_position'][2],
                'marker_rot_x': point['marker_rotation'][0],
                'marker_rot_y': point['marker_rotation'][1],
                'marker_rot_z': point['marker_rotation'][2],
            }
            
            # Add joint angles
            for i, angle in enumerate(point['joint_angles']):
                row[f'joint_{i+1}'] = angle
            
            # Add end effector pose if available
            if point['ee_position'] is not None:
                row.update({
                    'ee_pos_x': point['ee_position'][0],
                    'ee_pos_y': point['ee_position'][1],
                    'ee_pos_z': point['ee_position'][2],
                })
            
            if point['ee_orientation'] is not None:
                row.update({
                    'ee_rot_x': point['ee_orientation'][0],
                    'ee_rot_y': point['ee_orientation'][1],
                    'ee_rot_z': point['ee_orientation'][2],
                })
            
            # Add robot info
            robot_info = point.get('robot_info', {})
            row['robot_connected'] = robot_info.get('connected', False)
            row['bus_connected'] = robot_info.get('bus_connected', False)
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Data saved to:")
        print(f"   📄 JSON: {json_filename}")
        print(f"   📊 CSV:  {csv_filename}")
        print(f"   📈 Total data points: {len(self.collected_data)}")


def draw_instructions(frame, collector):
    """Draw instruction text on the frame"""
    instructions = [
        "SO101 Follower Calibration Data Collection",
        "Controls:",
        "SPACE - Collect data point",
        "1-9   - Change target marker ID (1-9)",
        "0     - Set target marker ID to 0", 
        "D     - Toggle debug info",
        "S     - Save data",
        "Q     - Quit",
        "",
        f"Target marker ID: {collector.target_marker_id}",
        f"Detected markers: {sorted(list(collector.detected_markers)) if collector.detected_markers else 'None'}"
    ]
    
    # Draw background for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (500, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    y_offset = 30
    for i, instruction in enumerate(instructions):
        if "Target marker ID:" in instruction:
            color = (0, 255, 255)  # Yellow for current target
        elif "Detected markers:" in instruction:
            color = (255, 255, 0)  # Cyan for detected markers
        elif i == 0:
            color = (0, 255, 255)  # Yellow for title
        else:
            color = (255, 255, 255)  # White for normal text
            
        font_scale = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        
        cv2.putText(frame, instruction, (20, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return frame


def generate_aruco_marker(marker_id=0, size=200):
    """Generate and save an ArUco marker for printing"""
    try:
        # Create ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
        
        # Save marker
        filename = f"aruco_marker_id_{marker_id}.png"
        cv2.imwrite(filename, marker_img)
        print(f"📄 ArUco marker ID {marker_id} saved as: {filename}")
        print(f"   Print this image and use it for calibration")
        return True
    except Exception as e:
        print(f"❌ Error generating ArUco marker: {e}")
        return False

def main():
    """Main data collection loop with camera"""
    print("🤖 SO101 Follower Real-time Calibration Data Collection")
    print("=" * 60)
    
    # Generate ArUco markers for user
    print("📄 Generating ArUco markers for calibration...")
    for marker_id in [0, 1, 2, 3, 4]:
        generate_aruco_marker(marker_id)
    print("✅ ArUco markers generated! Print them and hold in front of camera.")
    print()
    
    # Camera calibration parameters (you should calibrate your specific camera)
    # These are example values - replace with your actual camera calibration
    # Camera intrinsic matrix
    camera_matrix = np.array([[1101.4773366334907, 0.0, 642.6644842801771], [0.0, 1102.9541640653897, 380.20234515849927], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.array([-0.0846105588575387, 0.14179204930913328, 0.001127442817630244, -0.0017247298769922544, -0.06517732350331334], dtype=np.float32)
    marker_length = 0.10  # 5cm marker - adjust based on your actual marker size
    
    print("📋 Initializing components...")
    
    # Initialize components
    aruco_tracker = ArucoTracker(camera_matrix, dist_coeffs, marker_length)
    
    # Initialize robot interface with default port (modify as needed)
    robot_port = input("Enter robot port (press Enter for default '/dev/tty.usbmodem5A460827181'): ").strip()
    if not robot_port:
        robot_port = "/dev/tty.usbmodem5A460827181"
    
    robot_interface = SO101RobotInterface(port=robot_port)
    
    # Try to connect to robot
    if not robot_interface.connect():
        print("⚠️  Warning: Could not connect to robot. Data collection will continue without real robot data.")
        print("   Make sure the robot is connected and the port is correct.")
    
    collector = CalibrationDataCollector(aruco_tracker, robot_interface, target_marker_id=0)
    
    # Initialize camera
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        robot_interface.disconnect()
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully")
    print("\n🎥 Camera window controls:")
    print("   SPACE - Collect data point when target marker is visible")
    print("   1-9   - Change target marker ID to 1-9")
    print("   0     - Change target marker ID to 0")
    print("   D     - Toggle debug information")
    print("   S     - Save collected data")
    print("   Q     - Quit application")
    
    # Initialize FPS counter
    fps_counter = 0
    fps_start_time = time.time()
    show_debug = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Detect markers and draw
            poses, ids, corners = aruco_tracker.detect_and_estimate_pose(frame)
            
            # Update detected markers in collector
            if ids is not None:
                collector.detected_markers.update(ids.flatten().tolist())
            
            # Draw detected markers
            if ids is not None:
                try:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                except:
                    # Fallback for older OpenCV versions
                    for i, corner in enumerate(corners):
                        cv2.polylines(frame, [corner.astype(int)], True, (0, 255, 0), 2)
                        # Draw marker ID
                        center = np.mean(corner[0], axis=0).astype(int)
                        cv2.putText(frame, f"ID:{ids[i][0]}", tuple(center), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw poses and highlight target marker
                for i, (rvec, tvec) in enumerate(poses):
                    frame = aruco_tracker.draw_pose(frame, rvec, tvec)
                    
                    # Highlight target marker
                    if ids[i][0] == collector.target_marker_id:
                        # Draw a green circle around the target marker
                        center = np.mean(corners[i][0], axis=0).astype(int)
                        cv2.circle(frame, tuple(center), 50, (0, 255, 0), 3)
                        cv2.putText(frame, f"TARGET: {ids[i][0]}", 
                                  (center[0] - 40, center[1] - 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Show marker position
                        pos_text = f"Pos: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]"
                        cv2.putText(frame, pos_text, 
                                  (center[0] - 80, center[1] + 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw instructions overlay
            frame = draw_instructions(frame, collector)
            
            # Display collection info
            status_text = f"Collected: {len(collector.collected_data)} points"
            if 'fps' in locals():
                status_text += f" | FPS: {fps:.1f}"
            
            # Add robot status
            robot_status = "🟢 Connected" if robot_interface.is_connected else "🔴 Disconnected"
            status_text += f" | Robot: {robot_status}"
            
            cv2.putText(frame, status_text, 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show marker detection status
            if ids is not None and collector.target_marker_id in ids.flatten():
                cv2.putText(frame, f"✓ TARGET MARKER {collector.target_marker_id} DETECTED", 
                           (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"○ Looking for marker ID {collector.target_marker_id}...", 
                           (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show debug info if enabled
            if show_debug and ids is not None:
                debug_y = 100
                for i in range(len(ids)):
                    debug_text = f"ID {ids[i][0]}: corners={len(corners[i][0])}"
                    cv2.putText(frame, debug_text, (frame.shape[1] - 300, debug_y + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Display the frame
            cv2.imshow('SO101 Follower Calibration Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to collect data
                data_point = collector.collect_data_point(frame)
                if data_point:
                    collector.add_data_point(data_point)
                else:
                    print(f"❌ Target marker ID {collector.target_marker_id} not detected!")
                    if collector.detected_markers:
                        print(f"   Available markers: {sorted(list(collector.detected_markers))}")
                    else:
                        print("   No markers detected at all - check lighting and marker visibility")
                
            elif key >= ord('0') and key <= ord('9'):  # Change target marker ID
                new_id = key - ord('0')
                collector.update_target_marker(new_id)
                
            elif key == ord('d') or key == ord('D'):  # Toggle debug
                show_debug = not show_debug
                print(f"🔧 Debug mode: {'ON' if show_debug else 'OFF'}")
                
            elif key == ord('s') or key == ord('S'):  # Save data
                if collector.collected_data:
                    collector.save_data("so101_calibration_data")
                else:
                    print("❌ No data to save!")
            
            elif key == ord('q') or key == ord('Q'):  # Quit
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        robot_interface.disconnect()
        
        # Save data on exit if any collected
        if collector.collected_data:
            print(f"\n💾 Saving {len(collector.collected_data)} data points before exit...")
            collector.save_data("so101_calibration_data")
        
        print("\n✨ Data collection session completed!")
        print("📁 Check your working directory for saved data files.")


if __name__ == "__main__":
    # Set up logging to reduce verbosity
    logging.getLogger("lerobot").setLevel(logging.WARNING)
    
    main()