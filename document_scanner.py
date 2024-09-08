# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
from utils import load_image, save_image

def scan_document(image_path, image_name):
    print("Scanning document", image_name)
    image = load_image(image_path)
    
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    edged = find_edges(image)
    
    try:
        image_with_contours, screen_cnt = find_contours(edged, image)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Apply the four point transform to obtain a top-down
    # View of the original image
    warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255

    folder_name = 'results/' + image_name.split(".")[0]
    save_image(folder_name, '1_original_image.jpg', image)
    save_image(folder_name, '2_edged_image.jpg', edged)
    save_image(folder_name, '3_image_with_contour.jpg', image_with_contours)
    save_image(folder_name, '4_scanned_image.jpg', warped)
    print("Saved document", image_name)
    return warped

def find_edges(image):
	# load the image and resize it
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return edged


def find_contours(edged, image):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour

    # This finds the contours (i.e., the boundaries or outlines of shapes) in the image.
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # The cv2.findContours() function can return different outputs based on the OpenCV version, so imutils.grab_contours(cnts) is a helper function to standardize the format, making sure you only get the contour list.
    cnts = imutils.grab_contours(cnts)

    # Sorts the contours by area, keeping only the top 5 largest contours. You sort by cv2.contourArea to prioritize bigger contours, which are more likely to correspond to large objects like a piece of paper.
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # Loops through the top 5 contours and approximates each one to determine if it's a quadrilateral (i.e., a shape with 4 sides).
    for c in cnts:
        # Calculates the perimeter of the contour c.
        peri = cv2.arcLength(c, True)
        
        # Approximates the contour with fewer points based on the perimeter. This simplifies the contour and helps determine if it can be approximated as a rectangle (the piece of paper).
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        found_quad = False
        if len(approx) == 4:
            screen_cnt = approx
            found_quad = True
            break

    # show the contour (outline) of the piece of paper
    if not found_quad:
        raise ValueError("The image provided does not contain a paper, or the image is not clear.")

    cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
    return image, screen_cnt

