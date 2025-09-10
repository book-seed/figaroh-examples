#!/usr/bin/env python3
"""
ArUco Detection Status Monitor
Shows what the camera is detecting in real-time
"""

import cv2
import numpy as np
import time

def main():
    print("🔍 ArUco Detection Status Monitor")
    print("=" * 40)
    print("This will show you what ArUco markers are being detected")
    print("Press 'q' to quit")
    
    # Initialize ArUco detector
    try:
        # Try OpenCV 4.7+ API
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        print("✅ Using OpenCV 4.7+ ArUco API")
    except AttributeError:
        # Fallback to older API
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        detector = None
        print("✅ Using legacy ArUco API")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("📹 Camera opened successfully")
    print("🎯 Looking for ArUco markers...")
    
    frame_count = 0
    last_detection_time = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        if detector:  # OpenCV 4.7+
            corners, ids, _ = detector.detectMarkers(gray)
        else:  # Legacy API
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Draw detection info
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Update last detection time
            last_detection_time = time.time()
            
            # Show detection info
            detection_text = f"DETECTED: {len(ids)} markers"
            cv2.putText(frame, detection_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # List detected IDs
            ids_text = f"IDs: {[id[0] for id in ids]}"
            cv2.putText(frame, ids_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Highlight ID 0 if found
            if 0 in [id[0] for id in ids]:
                cv2.putText(frame, "🎯 TARGET ID 0 FOUND!", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # No markers detected
            cv2.putText(frame, "No markers detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show time since last detection
        if last_detection_time:
            time_since = time.time() - last_detection_time
            if time_since < 1.0:
                status_color = (0, 255, 0)  # Green - recent detection
                status_text = f"Last seen: {time_since:.1f}s ago"
            else:
                status_color = (0, 165, 255)  # Orange - older detection
                status_text = f"Last seen: {time_since:.1f}s ago"
        else:
            status_color = (0, 0, 255)  # Red - never detected
            status_text = "Never detected"
        
        cv2.putText(frame, status_text, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Instructions
        cv2.putText(frame, "Hold ArUco marker ID 0 in front of camera", 
                   (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('ArUco Detection Monitor', frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✨ Detection monitor closed")

if __name__ == "__main__":
    main()
