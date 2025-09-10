#!/usr/bin/env python3
"""
Real-time data collection with camera view and ArUco marker detection using SO101 robot motor bus directly
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

# Add the lerobot path to access the motor modules
sys.path.append('/Users/thanhndv212/Develop/lerobot')

from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus


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
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
            print("✅ Using OpenCV 4.7+ ArUco API")
        except AttributeError:
            # OpenCV 4.6 and earlier
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
            print("✅ Using OpenCV 4.6 ArUco API")
    
    def detect_and_estimate_pose(self, frame):
        """
        Detect ArUco markers and estimate their 6D pose
        
        Returns:
            poses: list of (rvec, tvec) tuples for each detected marker
            ids: list of marker IDs
            corners: detected marker corners
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers (OpenCV version compatible)
        if self.use_new_api:
            # OpenCV 4.7+
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            # OpenCV 4.6 and earlier
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        poses = []
        if ids is not None:
            # Estimate pose for each marker (OpenCV version compatible)
            try:
                # OpenCV 4.7+ - use solvePnP for each marker
                for i in range(len(corners)):
                    # Define 3D points of the marker corners in marker coordinate system
                    half_size = self.marker_length / 2
                    object_points = np.array([
                        [-half_size, half_size, 0],
                        [half_size, half_size, 0], 
                        [half_size, -half_size, 0],
                        [-half_size, -half_size, 0]
                    ], dtype=np.float32)
                    
                    # Solve PnP to get pose
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, corners[i], 
                        self.camera_matrix, self.dist_coeffs
                    )
                    
                    if success:
                        poses.append((rvec.flatten(), tvec.flatten()))
                    else:
                        poses.append((np.zeros(3), np.zeros(3)))
                        
            except Exception:
                # Fallback to older API if available
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                    )
                    for i in range(len(ids)):
                        poses.append((rvecs[i][0], tvecs[i][0]))
                except AttributeError:
                    # If estimatePoseSingleMarkers is not available, use manual solvePnP
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


class MotorBusRobotInterface:
    """Direct motor bus interface for SO101-like robots using Feetech motors"""
    
    def __init__(self, port: str = "/dev/tty.usbmodem575E0031751"):
        """
        Initialize the motor bus interface
        
        Args:
            port: Serial port for the motor communication
        """
        self.port = port
        
        # Define motors for SO101 Follower (Feetech STS3215)
        # Using normalized mode for easier handling
        norm_mode_body = MotorNormMode.RANGE_M100_100  # -100 to +100 range
        self.motors = {
            "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # 0-100 for gripper
        }
        
        # Initialize motor bus
        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=self.motors
        )
        
        self.is_connected = False
        self.motor_names = list(self.motors.keys())
        
        print(f"🤖 Motor Bus Interface initialized with port: {port}")
        print(f"   Motors: {list(self.motors.keys())}")
    
    def connect(self) -> bool:
        """Connect to the motor bus"""
        try:
            print("🔌 Connecting to motor bus...")
            self.bus.connect()
            self.is_connected = True
            print("✅ Motor bus connected successfully!")
            print(f"   Connected motors: {len(self.bus.motors)}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to motor bus: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the motor bus"""
        if self.is_connected:
            try:
                self.bus.disconnect()
                self.is_connected = False
                print("✅ Motor bus disconnected successfully!")
            except Exception as e:
                print(f"❌ Error disconnecting motor bus: {e}")
    
    def get_joint_angles(self) -> List[float]:
        """
        Get current joint angles from the motors
        
        Returns:
            List of joint positions (normalized values)
        """
        if not self.is_connected:
            print("❌ Motor bus not connected! Cannot read joint angles.")
            return [0.0] * len(self.motor_names)
        
        try:
            # Read motor positions using sync_read (this uses the Present_Position register)
            positions = self.bus.sync_read("Present_Position", normalize=True)
            
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
            return [0.0] * len(self.motor_names)
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end effector pose (simplified approximation)
        For a real system, this would use forward kinematics
        
        Returns:
            position: 3D position (x, y, z)
            orientation: 3D orientation (rx, ry, rz)
        """
        joint_angles = self.get_joint_angles()
        
        if len(joint_angles) >= 3:
            # Convert normalized values (-100 to 100) to approximate radians
            # This is a very simplified approximation
            shoulder_pan = joint_angles[0] * np.pi / 100.0  # Convert to radians
            shoulder_lift = joint_angles[1] * np.pi / 100.0
            elbow_flex = joint_angles[2] * np.pi / 100.0
            
            # Simple forward kinematics approximation
            # Note: These are rough estimates, real FK would need proper DH parameters
            L1 = 0.15  # Shoulder to elbow (approximate)
            L2 = 0.15  # Elbow to wrist (approximate)
            
            x = (L1 * np.cos(shoulder_lift) + L2 * np.cos(shoulder_lift + elbow_flex)) * np.cos(shoulder_pan)
            y = (L1 * np.cos(shoulder_lift) + L2 * np.cos(shoulder_lift + elbow_flex)) * np.sin(shoulder_pan)
            z = 0.1 + L1 * np.sin(shoulder_lift) + L2 * np.sin(shoulder_lift + elbow_flex)  # Base height offset
            
            position = np.array([x, y, z])
            
            # Orientation (simplified)
            wrist_flex = joint_angles[3] * np.pi / 100.0 if len(joint_angles) > 3 else 0.0
            wrist_roll = joint_angles[4] * np.pi / 100.0 if len(joint_angles) > 4 else 0.0
            orientation = np.array([wrist_flex, wrist_roll, shoulder_pan])
            
        else:
            position = np.array([0.2, 0.0, 0.2])
            orientation = np.array([0.0, 0.0, 0.0])
        
        return position, orientation
    
    def get_motor_info(self) -> Dict[str, Any]:
        """Get information about the motor state"""
        info = {
            "connected": self.is_connected,
            "motor_count": len(self.motor_names),
            "motor_names": self.motor_names,
            "port": self.port,
            "bus_connected": self.bus.is_connected if hasattr(self.bus, 'is_connected') else False
        }
        
        if self.is_connected:
            try:
                joint_angles = self.get_joint_angles()
                info["current_joints"] = joint_angles
                ee_pos, ee_ori = self.get_end_effector_pose()
                info["end_effector_position"] = ee_pos.tolist()
                info["end_effector_orientation"] = ee_ori.tolist()
                
                # Try to get additional motor info
                try:
                    # Read some motor status if possible
                    temperatures = self.bus.sync_read("Present_Temperature", normalize=False)
                    info["motor_temperatures"] = temperatures
                except:
                    pass
                    
            except Exception as e:
                info["error"] = str(e)
        
        return info


class CalibrationDataCollector:
    def __init__(self, aruco_tracker: ArucoTracker, robot_interface: MotorBusRobotInterface, target_marker_id: int = 0):
        """
        Initialize calibration data collector
        
        Args:
            aruco_tracker: ArUco marker tracker instance
            robot_interface: Motor bus robot interface
            target_marker_id: ID of the ArUco marker to track
        """
        self.aruco_tracker = aruco_tracker
        self.robot_interface = robot_interface
        self.target_marker_id = target_marker_id
        
        self.collected_data = []
        self.is_collecting = False
        
    def collect_data_point(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Collect a single data point (marker pose + joint angles)
        
        Returns:
            Dictionary containing the collected data or None if marker not detected
        """
        # Get ArUco marker pose
        poses, ids, corners = self.aruco_tracker.detect_and_estimate_pose(frame)
        
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
            'motor_info': self.robot_interface.get_motor_info()
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
        print(f"   Joints 1-3: [{joints[0]:.2f}, {joints[1]:.2f}, {joints[2]:.2f}] (normalized)")
        
        # Show motor connection status
        motor_info = data_point.get('motor_info', {})
        print(f"   Motors connected: {motor_info.get('connected', 'unknown')}")
        print(f"   Bus connected: {motor_info.get('bus_connected', 'unknown')}")
    
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
            
            # Add motor info
            motor_info = point.get('motor_info', {})
            row['motors_connected'] = motor_info.get('connected', False)
            row['bus_connected'] = motor_info.get('bus_connected', False)
            row['motor_count'] = motor_info.get('motor_count', 0)
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Data saved to:")
        print(f"   📄 JSON: {json_filename}")
        print(f"   📊 CSV:  {csv_filename}")
        print(f"   📈 Total data points: {len(self.collected_data)}")


def draw_instructions(frame):
    """Draw instruction text on the frame"""
    instructions = [
        "Motor Bus Calibration Data Collection",
        "Controls:",
        "SPACE - Collect data point",
        "S - Save data",
        "Q - Quit",
        "",
        f"ArUco markers detected: Look for ID 0"
    ]
    
    # Draw background for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    y_offset = 30
    for i, instruction in enumerate(instructions):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Yellow for title
        font_scale = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        
        cv2.putText(frame, instruction, (20, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return frame


def main():
    """Main data collection loop with camera"""
    print("🤖 Motor Bus Real-time Calibration Data Collection")
    print("=" * 55)
    
    # Camera calibration parameters (you should calibrate your specific camera)
    # These are example values - replace with your actual camera calibration
    camera_matrix = np.array([[800, 0, 320],
                             [0, 800, 240],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # assuming no distortion
    marker_length = 0.05  # 5cm marker - adjust based on your actual marker size
    
    print("📋 Initializing components...")
    
    # Initialize components
    aruco_tracker = ArucoTracker(camera_matrix, dist_coeffs, marker_length)
    
    # Initialize robot interface with default port (modify as needed)
    robot_port = input("Enter motor bus port (press Enter for default '/dev/tty.usbmodem575E0031751'): ").strip()
    if not robot_port:
        robot_port = "/dev/tty.usbmodem575E0031751"
    
    robot_interface = MotorBusRobotInterface(port=robot_port)
    
    # Try to connect to motor bus
    if not robot_interface.connect():
        print("⚠️  Warning: Could not connect to motor bus. Data collection will continue without real motor data.")
        print("   Make sure the motors are connected and the port is correct.")
    
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
    print("   SPACE - Collect data point when marker is visible")
    print("   S     - Save collected data")
    print("   Q     - Quit application")
    print("\n📋 Instructions:")
    print("   1. Hold an ArUco marker (ID 0) in front of the camera")
    print("   2. Press SPACE when the marker is clearly visible")
    print("   3. Move the robot and marker to different positions and collect more points")
    print("   4. Press S to save your data")
    print("\n🔴 Starting camera feed...")
    
    fps_counter = 0
    fps_start_time = time.time()
    
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
            
            # Draw detected markers
            if ids is not None:
                try:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                except:
                    # Fallback for older OpenCV versions
                    for i, corner in enumerate(corners):
                        cv2.polylines(frame, [corner.astype(int)], True, (0, 255, 0), 2)
                
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
            frame = draw_instructions(frame)
            
            # Display collection info
            status_text = f"Collected: {len(collector.collected_data)} points"
            if 'fps' in locals():
                status_text += f" | FPS: {fps:.1f}"
            
            # Add motor status
            motor_status = "🟢 Connected" if robot_interface.is_connected else "🔴 Disconnected"
            status_text += f" | Motors: {motor_status}"
            
            cv2.putText(frame, status_text, 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show marker detection status
            if ids is not None and collector.target_marker_id in ids.flatten():
                cv2.putText(frame, "✓ TARGET MARKER DETECTED", 
                           (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "○ Looking for marker ID 0...", 
                           (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Motor Bus Calibration Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to collect data
                data_point = collector.collect_data_point(frame)
                if data_point:
                    collector.add_data_point(data_point)
                else:
                    print("❌ Target marker not detected! Make sure marker ID 0 is visible.")
                
            elif key == ord('s') or key == ord('S'):  # Save data
                if collector.collected_data:
                    collector.save_data("motor_bus_calibration_data")
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
            collector.save_data("motor_bus_calibration_data")
        
        print("\n✨ Data collection session completed!")
        print("📁 Check your working directory for saved data files.")


if __name__ == "__main__":
    # Set up logging to reduce verbosity
    logging.getLogger("lerobot").setLevel(logging.WARNING)
    
    main()
