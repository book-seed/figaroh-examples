#!/usr/bin/env python3
"""
Real-time data collection with camera view and ArUco marker detection
"""

import cv2
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any


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


class MockRobotInterface:
    """Mock robot interface for testing"""
    
    def __init__(self):
        self.step = 0
    
    def get_joint_angles(self) -> List[float]:
        """Return mock joint angles that change over time"""
        self.step += 1
        base_angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # Add small variations to simulate movement
        return [angle + 0.01 * np.sin(self.step * 0.1 + i) for i, angle in enumerate(base_angles)]
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return mock end effector pose"""
        # Simulate slight movement
        x = 0.3 + 0.05 * np.sin(self.step * 0.05)
        y = 0.2 + 0.03 * np.cos(self.step * 0.07)
        z = 0.5 + 0.02 * np.sin(self.step * 0.03)
        
        position = np.array([x, y, z])
        orientation = np.array([0.0, 0.0, self.step * 0.01])  # Slowly rotating
        return position, orientation


class CalibrationDataCollector:
    def __init__(self, aruco_tracker: ArucoTracker, robot_interface: MockRobotInterface, target_marker_id: int = 0):
        """
        Initialize calibration data collector
        
        Args:
            aruco_tracker: ArUco marker tracker instance
            robot_interface: Robot communication interface
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
            'marker_id': int(self.target_marker_id)
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
        "FIGAROH Calibration Data Collection",
        "Controls:",
        "SPACE - Collect data point",
        "S - Save data",
        "Q - Quit",
        "",
        f"ArUco markers detected: Look for ID 0"
    ]
    
    # Draw background for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    y_offset = 30
    for i, instruction in enumerate(instructions):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Yellow for title
        font_scale = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        
        cv2.putText(frame, instruction, (20, y_offset + i * 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return frame


def main():
    """Main data collection loop with camera"""
    print("🤖 FIGAROH Real-time Calibration Data Collection")
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
    robot_interface = MockRobotInterface()  # Replace with actual robot interface
    collector = CalibrationDataCollector(aruco_tracker, robot_interface, target_marker_id=0)
    
    # Initialize camera
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
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
    print("   3. Move the marker to different positions and collect more points")
    print("   4. Press S to save your data")
    print("\n🔴 Starting camera feed...")
    
    fps_counter = 0
    fps_start_time = time.time()
    
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
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
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
        status_text = f"Collected: {len(collector.collected_data)} points | FPS: {fps:.1f}" if 'fps' in locals() else f"Collected: {len(collector.collected_data)} points"
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
        cv2.imshow('FIGAROH Calibration Data Collection', frame)
        
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
                collector.save_data("camera_calibration_data")
            else:
                print("❌ No data to save!")
        
        elif key == ord('q') or key == ord('Q'):  # Quit
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save data on exit if any collected
    if collector.collected_data:
        print(f"\n💾 Saving {len(collector.collected_data)} data points before exit...")
        collector.save_data("camera_calibration_data")
    
    print("\n✨ Data collection session completed!")
    print("📁 Check your working directory for saved data files.")


if __name__ == "__main__":
    main()
