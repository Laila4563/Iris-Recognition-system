import os
import pickle # used for saving results into file for later
import numpy as np
from preprocess import preprocess_image

# This function calculates the average Hue histogram for each individual based on their left and right eye images.
def compute_average_histograms(dataset_path):
    # Dictionary store avg histogram key: person's folder name and value: avg histo 1D array
    person_histograms = {} 
    
    # Iterate over each person's folder
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        
        all_histograms = []
        # loop over eyes, check if folders exists, iterate over images and preprocess
        for eye in ['Left_Eye', 'Right_Eye']:
            eye_path = os.path.join(person_path, eye)
            if os.path.exists(eye_path):
                for img_file in os.listdir(eye_path):
                    img_path = os.path.join(eye_path, img_file)
                    histogram = preprocess_image(img_path)
                    all_histograms.append(histogram)
        
        # Compute average histogram for the person
        average_histogram = np.mean(all_histograms, axis=0)
        person_histograms[person] = average_histogram
    
    # Saves the person_histograms dictionary to a file named average_histograms.pkl in binary format.
    # This allows for easy reloading later without recalculating the histograms.
    with open("average_histograms.pkl", "wb") as f:
        pickle.dump(person_histograms, f)
    print("Average histograms saved successfully!")

    # with open("average_histograms.pkl", "rb") as f:
    #      person_histograms = pickle.load(f)
    # print(person_histograms.keys())  # Should display the names of all persons
