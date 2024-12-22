import cv2 
import numpy as np
import os # used to interact with the file system (e.g., reading directories, handling paths).

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (256, 256))  # Resize 3shan kolo yekoon zy ba3d
    
    # Convert to HSV for better color representation.
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    
    # Focus on the Hue channel el heya awel channel fel HSV reprsenation
    hue_channel = hsv_img[:, :, 0]
    
    # Mask iris region using circular mask
    mask = np.zeros_like(hue_channel, dtype=np.uint8) # create a blank mask of the same size as hue channel
    center = (128, 128)  # Assume iris is centered in resized image
    cv2.circle(mask, center, 60, (255, 255, 255), -1)  # Mask size depends on image
    masked_hue = cv2.bitwise_and(hue_channel, hue_channel, mask=mask)
    
    # Calculate histogram
    hist = cv2.calcHist([masked_hue], [0], mask, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize the histogram

    return hist

