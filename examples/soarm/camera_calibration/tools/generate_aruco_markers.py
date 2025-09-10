#!/usr/bin/env python3
"""
Generate ArUco markers for calibration testing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_aruco_marker(marker_id=0, marker_size=200, save_path=None, display=True):
    """
    Generate an ArUco marker
    
    Args:
        marker_id: ID of the marker (0-249 for DICT_6X6_250)
        marker_size: Size of the marker in pixels
        save_path: Path to save the marker image (optional)
        display: Whether to display the marker
    
    Returns:
        marker_image: Generated marker as numpy array
    """
    
    # Create ArUco dictionary
    try:
        # OpenCV 4.7+
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print(f"✅ Using OpenCV 4.7+ ArUco API")
    except AttributeError:
        # OpenCV 4.6 and earlier
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        print(f"✅ Using OpenCV 4.6 ArUco API")
    
    # Generate marker (OpenCV version compatible)
    try:
        # OpenCV 4.7+
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    except AttributeError:
        # OpenCV 4.6 and earlier
        marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
    
    if save_path:
        cv2.imwrite(save_path, marker_image)
        print(f"💾 Marker saved to: {save_path}")
    
    if display:
        # Display using matplotlib for better visibility
        plt.figure(figsize=(6, 6))
        plt.imshow(marker_image, cmap='gray')
        plt.title(f'ArUco Marker ID: {marker_id}\nSize: {marker_size}x{marker_size} pixels')
        plt.axis('off')
        
        # Add instructions
        plt.figtext(0.5, 0.02, 
                   'Print this marker or display it on screen for calibration\n'
                   'Recommended physical size: 5cm x 5cm', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    return marker_image


def generate_marker_set(num_markers=4, marker_size=200, save_dir="aruco_markers"):
    """Generate a set of ArUco markers"""
    import os
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"📋 Generating {num_markers} ArUco markers...")
    
    for i in range(num_markers):
        marker_image = generate_aruco_marker(
            marker_id=i, 
            marker_size=marker_size, 
            save_path=f"{save_dir}/marker_{i:02d}.png",
            display=False
        )
    
    print(f"✅ Generated {num_markers} markers in '{save_dir}/' directory")
    
    # Create a combined image with all markers
    if num_markers <= 4:
        rows = 2
        cols = 2
    elif num_markers <= 9:
        rows = 3
        cols = 3
    else:
        rows = 4
        cols = 4
    
    combined_size = marker_size + 50  # Add padding
    combined_image = np.ones((rows * combined_size, cols * combined_size), dtype=np.uint8) * 255
    
    for i in range(min(num_markers, rows * cols)):
        row = i // cols
        col = i % cols
        
        # Generate marker (OpenCV version compatible)
        try:
            # OpenCV 4.7+
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            marker = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
        except AttributeError:
            # OpenCV 4.6 and earlier
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            marker = cv2.aruco.drawMarker(aruco_dict, i, marker_size)
        
        # Place marker in combined image
        y_start = row * combined_size + 25
        x_start = col * combined_size + 25
        combined_image[y_start:y_start+marker_size, x_start:x_start+marker_size] = marker
        
        # Add ID label
        cv2.putText(combined_image, f"ID: {i}", 
                   (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    # Save combined image
    combined_path = f"{save_dir}/combined_markers.png"
    cv2.imwrite(combined_path, combined_image)
    print(f"📄 Combined marker sheet saved to: {combined_path}")
    
    return combined_image


def main():
    """Main function to generate markers"""
    print("🎯 ArUco Marker Generator for FIGAROH Calibration")
    print("=" * 50)
    
    # Generate single marker for testing (ID 0)
    print("\n📋 Generating test marker (ID 0)...")
    marker = generate_aruco_marker(
        marker_id=0, 
        marker_size=400,  # Larger for printing
        save_path="aruco_marker_0.png",
        display=True
    )
    
    print("\n📋 Generating marker set...")
    combined = generate_marker_set(num_markers=4, marker_size=200)
    
    print("\n✨ Marker generation completed!")
    print("\n📋 Usage instructions:")
    print("1. Print 'aruco_marker_0.png' on paper (recommended size: 5cm x 5cm)")
    print("2. Or display it on a tablet/phone screen")
    print("3. Use the printed/displayed marker with the camera calibration script")
    print("4. Make sure the marker is well-lit and clearly visible to the camera")
    
    # Display the combined markers too
    plt.figure(figsize=(10, 10))
    plt.imshow(combined, cmap='gray')
    plt.title('ArUco Marker Set for Calibration')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
