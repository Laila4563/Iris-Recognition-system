import os
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocess import preprocess_image
from test import recognize_person

# Path to the test dataset
test_dataset_path = r"C:\Users\20111\OneDrive\Desktop\Image Processing\Iris recognition project\Test"  # Update with the actual test dataset path
histograms_file = "average_histograms.pkl"  # Trained histograms file

# Load trained average histograms
if not os.path.exists(histograms_file):
    raise FileNotFoundError("Model not trained yet! Please train the model first.")

with open(histograms_file, "rb") as f:
    person_histograms = pickle.load(f)

# Collect true labels and predicted labels
true_labels = []
predicted_labels = []

for person in os.listdir(test_dataset_path):
    person_path = os.path.join(test_dataset_path, person)
    if not os.path.isdir(person_path):
        continue
    
    for eye in ['Left_Eye', 'Right_Eye']:
        eye_path = os.path.join(person_path, eye)
        if os.path.exists(eye_path):
            for img_file in os.listdir(eye_path):
                img_path = os.path.join(eye_path, img_file)
                true_labels.append(person)  # Add the true label (folder name)
                
                # Predict using the recognize_person function
                predicted_person, _ = recognize_person(img_path, histograms_file)
                predicted_labels.append(predicted_person)

# Compute confusion matrix and accuracy
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=list(person_histograms.keys()))
accuracy = accuracy_score(true_labels, predicted_labels)

# Print results
print("Confusion Matrix:")
person_names = list(person_histograms.keys())  # Get the person names
for i, row in enumerate(conf_matrix):
    row_str = " ".join([f"{value}" for value in row])
    print(f"{person_names[i]}: {row_str}")
    
print(f"Accuracy: {accuracy:.2f}")
