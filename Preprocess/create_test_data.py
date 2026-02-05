#!/usr/bin/env python3
"""
Create test data for reconstruct_sem.py testing
"""

import numpy as np
import tifffile
import os

def create_test_data():
    """Create test data with shape (10, 100, 100)"""
    print("Creating test data...")
    
    # Create test data with shape (10, 100, 100)
    # Each slice will have different patterns for easy identification
    test_data = np.zeros((10, 100, 100), dtype=np.uint8)
    
    for i in range(10):
        # Create different patterns for each slice
        # Slice 0: horizontal lines
        if i == 0:
            test_data[i, ::10, :] = 255  # Every 10th row
        
        # Slice 1: vertical lines  
        elif i == 1:
            test_data[i, :, ::10] = 255  # Every 10th column
            
        # Slice 2: diagonal pattern
        elif i == 2:
            for j in range(100):
                if j < 100:
                    test_data[i, j, j] = 255
                    
        # Slice 3: checkerboard pattern
        elif i == 3:
            test_data[i, ::20, ::20] = 255
            test_data[i, 10::20, 10::20] = 255
            
        # Slice 4: gradient
        elif i == 4:
            test_data[i, :, :] = np.linspace(0, 255, 100).astype(np.uint8)
            
        # Slice 5: circle
        elif i == 5:
            center = 50
            y, x = np.ogrid[:100, :100]
            mask = (x - center)**2 + (y - center)**2 <= 20**2
            test_data[i][mask] = 255
            
        # Slice 6: random pattern
        elif i == 6:
            test_data[i] = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            
        # Slice 7: cross pattern
        elif i == 7:
            test_data[i, 45:55, :] = 255  # Horizontal line
            test_data[i, :, 45:55] = 255  # Vertical line
            
        # Slice 8: border
        elif i == 8:
            test_data[i, 0, :] = 255  # Top border
            test_data[i, -1, :] = 255  # Bottom border
            test_data[i, :, 0] = 255  # Left border
            test_data[i, :, -1] = 255  # Right border
            
        # Slice 9: filled rectangle
        elif i == 9:
            test_data[i, 30:70, 30:70] = 255
    
    # Save test data
    test_file = "test_drift_merged.tif"
    tifffile.imwrite(test_file, test_data)
    
    print(f"Test data saved as: {test_file}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data dtype: {test_data.dtype}")
    print(f"Test data range: {test_data.min()} - {test_data.max()}")
    
    return test_file

if __name__ == "__main__":
    create_test_data()





