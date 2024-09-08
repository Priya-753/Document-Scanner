from document_scanner import scan_document
import os
import cv2

def load_images_and_process(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    for image_file in image_files:
        # Build full path to the image
        image_path = os.path.join(folder_path, image_file)
        
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        print(f"Processing image: {image_file}")
        
        # Call your document_scan method to process the image
        scan_document(image_path, image_file)

# Folder containing the images
image_folder = 'images'

# Load images and process them one by one
load_images_and_process(image_folder)