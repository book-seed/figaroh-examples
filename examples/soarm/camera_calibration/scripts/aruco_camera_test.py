#!/usr/bin/env python3
"""
Simplified ArUco marker detection and pose estimation script for testing
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional

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


def draw_instructions(frame, target_marker_id, detected_markers):
    """Draw instruction text on the frame"""
    instructions = [
        "ArUco Marker Detection & Pose Estimation Test",
        "Controls:",
        "1-9   - Change target marker ID (1-9)",
        "0     - Set target marker ID to 0", 
        "Q     - Quit",
        "",
        f"Target marker ID: {target_marker_id}",
        f"Detected markers: {sorted(list(detected_markers)) if detected_markers else 'None'}"
    ]
    
    # Draw background for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
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


def main():
    """Main camera loop with ArUco detection"""
    print("🎯 ArUco Marker Detection & Pose Estimation Test")
    print("=" * 50)
    
    # Generate ArUco markers for user
    print("📄 Generating ArUco markers for testing...")
    for marker_id in [0, 1, 2, 3, 4]:
        generate_aruco_marker(marker_id)
    print("✅ ArUco markers generated! Print them and hold in front of camera.")
    print()
    
    # Camera calibration parameters (you should calibrate your specific camera)
    # These are example values - replace with your actual camera calibration
    camera_matrix = np.array([
        [1101.4773366334907, 0.0, 642.6644842801771], 
        [0.0, 1102.9541640653897, 380.20234515849927], 
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.array([
        -0.0846105588575387, 0.14179204930913328, 0.001127442817630244, 
        -0.0017247298769922544, -0.06517732350331334
    ], dtype=np.float32)
    
    marker_length = 0.05  # 5cm marker - adjust based on your actual marker size
    
    print("📋 Initializing ArUco tracker...")
    aruco_tracker = ArucoTracker(camera_matrix, dist_coeffs, marker_length)
    
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
    
    # Wait a moment for camera to initialize
    time.sleep(1)
    
    # Test frame capture
    test_ret, test_frame = cap.read()
    if not test_ret:
        print("❌ Error: Could not read test frame from camera")
        cap.release()
        return
    
    print("✅ Camera initialized successfully")
    print(f"   Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print("\n🎥 Camera window controls:")
    print("   1-9   - Change target marker ID to 1-9")
    print("   0     - Change target marker ID to 0")
    print("   Q     - Quit application")
    
    # Initialize tracking variables
    target_marker_id = 0
    detected_markers = set()
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
            
            # Update detected markers
            if ids is not None:
                detected_markers.update(ids.flatten().tolist())
            
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
                    if ids[i][0] == target_marker_id:
                        # Draw a green circle around the target marker
                        center = np.mean(corners[i][0], axis=0).astype(int)
                        cv2.circle(frame, tuple(center), 50, (0, 255, 0), 3)
                        cv2.putText(frame, f"TARGET: {ids[i][0]}", 
                                  (center[0] - 40, center[1] - 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Show marker position and rotation
                        pos_text = f"Pos: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]"
                        rot_text = f"Rot: [{rvec[0]:.3f}, {rvec[1]:.3f}, {rvec[2]:.3f}]"
                        cv2.putText(frame, pos_text, 
                                  (center[0] - 80, center[1] + 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(frame, rot_text, 
                                  (center[0] - 80, center[1] + 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw instructions overlay
            frame = draw_instructions(frame, target_marker_id, detected_markers)
            
            # Display FPS and detection status
            status_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"
            cv2.putText(frame, status_text, 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show marker detection status
            if ids is not None and target_marker_id in ids.flatten():
                cv2.putText(frame, f"✓ TARGET MARKER {target_marker_id} DETECTED", 
                           (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"○ Looking for marker ID {target_marker_id}...", 
                           (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('ArUco Marker Detection & Pose Estimation', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key >= ord('0') and key <= ord('9'):  # Change target marker ID
                target_marker_id = key - ord('0')
                print(f"🎯 Target marker updated to ID: {target_marker_id}")
                
            elif key == ord('q') or key == ord('Q'):  # Quit
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✨ ArUco detection test completed!")
        if detected_markers:
            print(f"📊 Detected marker IDs: {sorted(list(detected_markers))}")
        else:
            print("📊 No markers were detected during the session")


if __name__ == "__main__":
    main()
