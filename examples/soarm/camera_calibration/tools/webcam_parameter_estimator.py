#!/usr/bin/env python3
"""
Camera Parameter Estimator for Common Webcams
Provides better default parameters when exact calibration isn't available
"""

import numpy as np
import cv2


class WebcamParameterEstimator:
    """Estimate camera parameters for common webcam resolutions"""
    
    @staticmethod
    def get_typical_parameters(resolution: tuple = (1280, 720), fov_degrees: float = 60.0):
        """
        Get typical camera parameters for common webcam specifications
        
        Args:
            resolution: (width, height) of camera resolution
            fov_degrees: Horizontal field of view in degrees (typical: 50-80°)
            
        Returns:
            tuple: (camera_matrix, dist_coeffs)
        """
        width, height = resolution
        
        # Calculate focal length from field of view
        # fx = width / (2 * tan(fov/2))
        fov_rad = np.radians(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        
        # Assume square pixels (fy = fx) - common for modern cameras
        fy = fx
        
        # Principal point usually near image center
        cx = width / 2.0
        cy = height / 2.0
        
        # Camera matrix
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Typical distortion for basic webcams (minimal distortion)
        # Most modern webcams have lens correction built-in
        dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return camera_matrix, dist_coeffs
    
    @staticmethod
    def get_webcam_presets():
        """Get predefined parameters for common webcam types"""
        
        presets = {
            "basic_720p": {
                "resolution": (1280, 720),
                "fov": 60.0,
                "description": "Basic 720p webcam (most common)"
            },
            "wide_720p": {
                "resolution": (1280, 720), 
                "fov": 78.0,
                "description": "Wide-angle 720p webcam"
            },
            "basic_1080p": {
                "resolution": (1920, 1080),
                "fov": 65.0,
                "description": "Basic 1080p webcam"
            },
            "wide_1080p": {
                "resolution": (1920, 1080),
                "fov": 78.0,
                "description": "Wide-angle 1080p webcam"
            },
            "ultrawide": {
                "resolution": (1280, 720),
                "fov": 90.0,
                "description": "Ultra-wide webcam"
            }
        }
        
        result = {}
        for name, params in presets.items():
            camera_matrix, dist_coeffs = WebcamParameterEstimator.get_typical_parameters(
                params["resolution"], params["fov"]
            )
            result[name] = {
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "resolution": params["resolution"],
                "fov": params["fov"],
                "description": params["description"]
            }
        
        return result
    
    @staticmethod
    def detect_camera_resolution(camera_id: int = 0):
        """
        Detect actual camera resolution
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            tuple: (width, height) or None if failed
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None
        
        # Try to set high resolution and see what we actually get
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            cap.release()
            return (width, height)
        
        cap.release()
        return None
    
    @staticmethod
    def estimate_fov_from_detection(camera_id: int = 0, known_object_size: float = 0.1, 
                                   known_distance: float = 0.5):
        """
        Estimate field of view by measuring a known object at known distance
        
        Args:
            camera_id: Camera device ID
            known_object_size: Size of object in meters (e.g., 10cm = 0.1m)
            known_distance: Distance to object in meters (e.g., 50cm = 0.5m)
            
        Returns:
            float: Estimated horizontal FOV in degrees
        """
        print(f"📐 FOV Estimation Setup:")
        print(f"   1. Place an object of size {known_object_size*100:.1f}cm")
        print(f"   2. Position it {known_distance*100:.1f}cm from camera")
        print(f"   3. Measure its width in pixels")
        print(f"   4. Press 'q' to quit measurement")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            
            # Draw crosshairs for measurement
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 1)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 1)
            
            # Instructions
            cv2.putText(frame, f"Measure object width in pixels", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Object: {known_object_size*100:.1f}cm at {known_distance*100:.1f}cm distance", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('FOV Estimation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Get user input for measured pixel width
        try:
            pixel_width = float(input(f"Enter measured object width in pixels: "))
            
            # Calculate FOV
            # Angular size = 2 * arctan(object_size / (2 * distance))
            angular_size_rad = 2 * np.arctan(known_object_size / (2 * known_distance))
            
            # Pixel to angle ratio
            pixels_per_radian = pixel_width / angular_size_rad
            
            # Full frame FOV
            fov_rad = width / pixels_per_radian
            fov_degrees = np.degrees(fov_rad)
            
            return fov_degrees
            
        except ValueError:
            return None


def auto_detect_camera_parameters(camera_id: int = 0):
    """
    Automatically detect and suggest camera parameters
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        dict: Suggested camera parameters
    """
    print("🔍 Auto-detecting camera parameters...")
    
    # Detect resolution
    resolution = WebcamParameterEstimator.detect_camera_resolution(camera_id)
    if resolution is None:
        print("❌ Could not detect camera resolution")
        return None
    
    print(f"📊 Detected resolution: {resolution[0]} x {resolution[1]}")
    
    # Get presets
    presets = WebcamParameterEstimator.get_webcam_presets()
    
    # Find best matching preset
    best_match = None
    for name, preset in presets.items():
        if preset["resolution"] == resolution:
            if best_match is None or preset["fov"] == 60.0:  # Prefer 60° as most common
                best_match = (name, preset)
    
    if best_match is None:
        # Use generic parameters for detected resolution
        camera_matrix, dist_coeffs = WebcamParameterEstimator.get_typical_parameters(resolution)
        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "resolution": resolution,
            "fov": 60.0,
            "description": f"Generic parameters for {resolution[0]}x{resolution[1]}"
        }
    
    name, preset = best_match
    print(f"✅ Best match: {name} - {preset['description']}")
    
    return preset


def print_parameters_summary(params: dict):
    """Print camera parameters in various formats"""
    
    camera_matrix = params["camera_matrix"]
    dist_coeffs = params["dist_coeffs"]
    
    print("\n" + "="*60)
    print("📷 ESTIMATED CAMERA PARAMETERS")
    print("="*60)
    print(f"Description: {params['description']}")
    print(f"Resolution: {params['resolution'][0]} x {params['resolution'][1]}")
    print(f"Field of View: {params['fov']:.1f}°")
    
    print(f"\n🎯 Camera Matrix:")
    print(f"   fx = {camera_matrix[0,0]:.2f}")
    print(f"   fy = {camera_matrix[1,1]:.2f}")
    print(f"   cx = {camera_matrix[0,2]:.2f}")
    print(f"   cy = {camera_matrix[1,2]:.2f}")
    
    print(f"\n🔧 Distortion Coefficients:")
    print(f"   {dist_coeffs.flatten()}")
    
    print(f"\n📝 For FIGAROH camera_data_collection.py:")
    print(f"camera_matrix = np.array({camera_matrix.tolist()}, dtype=np.float32)")
    print(f"dist_coeffs = np.array({dist_coeffs.flatten().tolist()}, dtype=np.float32)")
    
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Webcam Parameter Estimator")
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--preset', choices=['basic_720p', 'wide_720p', 'basic_1080p', 'wide_1080p', 'ultrawide'],
                       help='Use specific preset')
    parser.add_argument('--fov', type=float, help='Manual FOV in degrees')
    parser.add_argument('--resolution', type=str, help='Manual resolution as "width,height"')
    
    args = parser.parse_args()
    
    if args.preset:
        # Use specific preset
        presets = WebcamParameterEstimator.get_webcam_presets()
        if args.preset in presets:
            params = presets[args.preset]
            print_parameters_summary(params)
        else:
            print(f"❌ Unknown preset: {args.preset}")
        return
    
    if args.resolution and args.fov:
        # Manual parameters
        try:
            width, height = map(int, args.resolution.split(','))
            camera_matrix, dist_coeffs = WebcamParameterEstimator.get_typical_parameters(
                (width, height), args.fov
            )
            params = {
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "resolution": (width, height),
                "fov": args.fov,
                "description": f"Manual parameters: {width}x{height} @ {args.fov}° FOV"
            }
            print_parameters_summary(params)
        except ValueError:
            print("❌ Invalid resolution format. Use 'width,height'")
        return
    
    # Auto-detection
    params = auto_detect_camera_parameters(args.camera)
    if params:
        print_parameters_summary(params)
        
        # Ask if user wants to save to FIGAROH
        save = input("\n💾 Save these parameters to FIGAROH camera_data_collection.py? (y/n): ")
        if save.lower() == 'y':
            update_figaroh_parameters(params)


def update_figaroh_parameters(params: dict):
    """Update camera parameters in FIGAROH camera_data_collection.py"""
    
    figaroh_file = "/Users/thanhndv212/Develop/figaroh/examples/soarm/camera_data_collection.py"
    
    try:
        # Read current file
        with open(figaroh_file, 'r') as f:
            content = f.read()
        
        # Find and replace camera matrix
        old_camera_matrix = """camera_matrix = np.array([[800, 0, 320],
                            [0, 800, 240], 
                            [0, 0, 1]], dtype=np.float32)"""
        
        new_camera_matrix = f"camera_matrix = np.array({params['camera_matrix'].tolist()}, dtype=np.float32)"
        
        # Find and replace distortion coefficients
        old_dist_coeffs = "dist_coeffs = np.zeros((4, 1))"
        new_dist_coeffs = f"dist_coeffs = np.array({params['dist_coeffs'].flatten().tolist()}, dtype=np.float32)"
        
        # Replace
        content = content.replace(old_camera_matrix, new_camera_matrix)
        content = content.replace(old_dist_coeffs, new_dist_coeffs)
        
        # Write back
        with open(figaroh_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated camera parameters in {figaroh_file}")
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")


if __name__ == "__main__":
    main()
