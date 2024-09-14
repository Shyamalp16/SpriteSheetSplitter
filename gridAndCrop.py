import cv2
import numpy as np
import os

def detect_sprite_sheet_grid(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Draw detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Save the result
    cv2.imwrite(output_path, image)
    
    print(f"Grid detection complete. Result saved as {output_path}")

def crop_sprite_cells(input_image, output_folder, min_sprite_size=100):
    # Read the image
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to separate sprites from background
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours of the sprites
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_sprite_size:
            x, y, w, h = cv2.boundingRect(contour)
            valid_contours.append((x, y, w, h, contour))
    
    # Sort contours based on their position (top to bottom, left to right)
    valid_contours.sort(key=lambda c: (c[1], c[0]))
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Crop and save each sprite
    for i, (x, y, w, h, contour) in enumerate(valid_contours):
        # Add padding
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Crop the sprite
        sprite = image[y:y+h, x:x+w]
        
        # Save the cropped sprite
        sprite_filename = f"sprite_{i}.png"
        sprite_path = os.path.join(output_folder, sprite_filename)
        cv2.imwrite(sprite_path, sprite)
    
    print(f"Cropped {len(valid_contours)} sprites and saved them in {output_folder}")

# Use the function
input_image = 'im1.png'  # Replace with your image path
output_image = 'grid6.png'
detect_sprite_sheet_grid(input_image, output_image)
crop_sprite_cells(input_image, 'cropped')