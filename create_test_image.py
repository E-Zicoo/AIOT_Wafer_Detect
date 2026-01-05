#!/usr/bin/env python3
"""
Generate a test wafer image with simulated defects for testing the wafer detect tool.
"""

import cv2
import numpy as np

def create_test_wafer_image(output_path='test_wafer.jpg', width=800, height=800):
    """
    Create a test wafer image with simulated defects.
    
    Args:
        output_path: Path to save the test image
        width: Image width in pixels
        height: Image height in pixels
    """
    # Create a white background (clean wafer)
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Draw circular wafer boundary
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 50
    cv2.circle(image, center, radius, (220, 220, 220), -1)
    cv2.circle(image, center, radius, (180, 180, 180), 3)
    
    # Add some simulated defects (dark spots)
    defects = [
        ((250, 200), 30),   # (center, radius)
        ((550, 300), 25),
        ((400, 500), 35),
        ((600, 600), 20),
        ((200, 600), 28),
    ]
    
    for defect_center, defect_radius in defects:
        # Create dark circular defects
        cv2.circle(image, defect_center, defect_radius, (50, 50, 50), -1)
        # Add some noise around the defect
        cv2.circle(image, defect_center, defect_radius + 5, (100, 100, 100), 2)
    
    # Add some smaller random defects
    np.random.seed(42)
    for _ in range(8):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        # Check if within wafer boundary
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        if dist_from_center < radius - 50:
            small_radius = np.random.randint(8, 15)
            cv2.circle(image, (x, y), small_radius, (60, 60, 60), -1)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Test wafer image created: {output_path}")
    print(f"Image size: {width}x{height} pixels")
    print(f"Simulated defects: {len(defects) + 8} (approximately)")

if __name__ == '__main__':
    create_test_wafer_image()
