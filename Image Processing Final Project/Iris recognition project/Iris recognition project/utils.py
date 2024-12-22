import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_histogram(image_path, title="RGB Histogram (Without Black Pixels)"):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image from BGR (OpenCV's default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a mask to exclude black pixels (where R, G, and B are all 0)
    mask = cv2.inRange(img_rgb, (1, 1, 1), (254, 254, 254))  # Exclude all-zero pixels

    # Calculate the histograms for each of the RGB channels using the mask
    r_hist = cv2.calcHist([img_rgb], [0], mask, [256], [0, 256])
    g_hist = cv2.calcHist([img_rgb], [1], mask, [256], [0, 256])
    b_hist = cv2.calcHist([img_rgb], [2], mask, [256], [0, 256])

    # Normalize the histograms to scale them between 0 and 1
    r_hist = cv2.normalize(r_hist, r_hist).flatten()
    g_hist = cv2.normalize(g_hist, g_hist).flatten()
    b_hist = cv2.normalize(b_hist, b_hist).flatten()

    # Plot the histograms
    plt.figure(figsize=(10, 6))
    plt.plot(r_hist, color='red', label='Red')
    plt.plot(g_hist, color='green', label='Green')
    plt.plot(b_hist, color='blue', label='Blue')
    
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


