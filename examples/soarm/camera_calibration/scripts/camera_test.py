#!/usr/bin/env python3
"""
Simple camera test to debug camera access issues
"""

import cv2
import time

def test_camera():
    print("🎥 Testing camera access...")
    
    # Try different camera indices
    for camera_idx in range(5):
        print(f"Trying camera index {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            print(f"✅ Camera {camera_idx} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Successfully read frame from camera {camera_idx}")
                print(f"   Frame shape: {frame.shape}")
                
                # Show the frame for 3 seconds
                cv2.imshow(f'Camera {camera_idx} Test', frame)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                
                cap.release()
                return camera_idx
            else:
                print(f"❌ Could not read frame from camera {camera_idx}")
        else:
            print(f"❌ Could not open camera {camera_idx}")
        
        cap.release()
    
    print("❌ No working cameras found")
    return None

def test_camera_properties(camera_idx):
    print(f"\n📊 Testing camera {camera_idx} properties...")
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Current resolution: {int(width)}x{int(height)}")
    print(f"Current FPS: {fps}")
    
    # Try to set properties
    print("Setting resolution to 640x480...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Verify settings
    new_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"New resolution: {int(new_width)}x{int(new_height)}")
    
    # Test frame capture
    print("Testing frame capture...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}: ✅ Success - shape: {frame.shape}")
        else:
            print(f"  Frame {i+1}: ❌ Failed")
        time.sleep(0.1)
    
    cap.release()

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        test_camera_properties(working_camera)
