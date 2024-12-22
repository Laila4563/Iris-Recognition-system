import cv2
import pickle
from preprocess import preprocess_image

def recognize_person(test_image, histograms_file):
    # Load stored average histograms with read mode
    with open(histograms_file, "rb") as f:
        person_histograms = pickle.load(f) # load dictionary of average histograms
    
    # Processes the input test image to extract the histogram for the iris region.
    test_histogram = preprocess_image(test_image)
    
    # Compare with each person's average histogram to find best match from file
    best_match = None
    highest_similarity = -1
    for person, avg_histogram in person_histograms.items():
        similarity = cv2.compareHist(test_histogram, avg_histogram, cv2.HISTCMP_CORREL)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = person
    
    return best_match, highest_similarity


























# over all steps 
# Load data 
# Preprocess the input
# Compare 
# Find best match 
# Return