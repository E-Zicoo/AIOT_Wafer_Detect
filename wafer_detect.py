#!/usr/bin/env python3
"""
Simple Wafer Detect Tool - Command Line Interface
Detects defects on semiconductor wafer images using image processing techniques.
"""

import argparse
import sys
import os
import cv2
import numpy as np


class WaferDetector:
    """Main class for wafer defect detection."""
    
    def __init__(self, threshold=127, min_defect_area=50):
        """
        Initialize the wafer detector.
        
        Args:
            threshold: Binary threshold value (0-255)
            min_defect_area: Minimum area in pixels to consider as a defect
        """
        self.threshold = threshold
        self.min_defect_area = min_defect_area
    
    def load_image(self, image_path):
        """
        Load an image from the given path.
        
        Args:
            image_path: Path to the wafer image file
            
        Returns:
            Loaded image or None if failed
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image: {image_path}")
            return None
        
        return image
    
    def preprocess_image(self, image):
        """
        Preprocess the wafer image for defect detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_defects(self, image):
        """
        Detect defects on the wafer image.
        
        Assumes defects are darker than the background. The detection uses
        THRESH_BINARY_INV, so darker regions will be identified as potential defects.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary containing detection results
        """
        # Preprocess the image
        preprocessed = self.preprocess_image(image)
        
        # Apply binary threshold
        _, binary = cv2.threshold(preprocessed, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (potential defects)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter defects by minimum area
        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_defect_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                # Get center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                defects.append({
                    'area': area,
                    'center': (cx, cy),
                    'bbox': (x, y, w, h),
                    'contour': contour
                })
        
        return {
            'defect_count': len(defects),
            'defects': defects,
            'binary_image': binary
        }
    
    def visualize_results(self, image, results, output_path=None):
        """
        Visualize detection results on the image.
        
        Args:
            image: Original BGR image
            results: Detection results dictionary
            output_path: Path to save the visualization (optional)
            
        Returns:
            Annotated image
        """
        # Create a copy for annotation
        annotated = image.copy()
        
        # Draw defects
        for defect in results['defects']:
            x, y, w, h = defect['bbox']
            cx, cy = defect['center']
            area = defect['area']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw center point
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            
            # Add area label
            cv2.putText(annotated, f"Area: {int(area)}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add summary text
        summary = f"Defects Found: {results['defect_count']}"
        cv2.putText(annotated, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Visualization saved to: {output_path}")
        
        return annotated


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Simple Wafer Detect Tool - Detect defects on semiconductor wafer images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg
  %(prog)s input.jpg -o output.jpg
  %(prog)s input.jpg -t 100 -m 30 -o result.jpg
        """
    )
    
    parser.add_argument('input', help='Input wafer image path')
    parser.add_argument('-o', '--output', help='Output image path for visualization')
    parser.add_argument('-t', '--threshold', type=int, default=127,
                       help='Binary threshold value (0-255, default: 127)')
    parser.add_argument('-m', '--min-area', type=int, default=50,
                       help='Minimum defect area in pixels (default: 50)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 255:
        print("Error: Threshold must be between 0 and 255")
        return 1
    
    # Validate min area
    if args.min_area < 0:
        print("Error: Minimum area must be non-negative")
        return 1
    
    # Create detector
    detector = WaferDetector(threshold=args.threshold, min_defect_area=args.min_area)
    
    if args.verbose:
        print(f"Loading image: {args.input}")
    
    # Load image
    image = detector.load_image(args.input)
    if image is None:
        return 1
    
    if args.verbose:
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"Detecting defects (threshold={args.threshold}, min_area={args.min_area})...")
    
    # Detect defects
    results = detector.detect_defects(image)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Wafer Defect Detection Results")
    print(f"{'='*50}")
    print(f"Total defects found: {results['defect_count']}")
    
    if results['defect_count'] > 0:
        print(f"\nDefect Details:")
        for i, defect in enumerate(results['defects'], 1):
            cx, cy = defect['center']
            area = defect['area']
            print(f"  Defect {i}: Center=({cx}, {cy}), Area={int(area)} pixels")
    
    print(f"{'='*50}\n")
    
    # Visualize results if output path is provided
    if args.output:
        detector.visualize_results(image, results, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
