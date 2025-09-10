#!/usr/bin/env python3
"""
Simple camera window test to verify OpenCV display functionality
"""

import cv2
import time

def main():
    print("🎥 Testing camera window display...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    print("📺 Camera window should appear now...")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Add simple text overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Camera Test Window', frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Auto-quit after 10 seconds for testing
        if frame_count > 100000:  # 30 FPS * 10 seconds
            print("🕐 Auto-quitting after 10 seconds")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✨ Camera test completed")

if __name__ == "__main__":
    main()
