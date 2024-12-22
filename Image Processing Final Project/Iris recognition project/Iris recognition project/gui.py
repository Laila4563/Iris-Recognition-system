import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from train import compute_average_histograms
from test import recognize_person
from preprocess import preprocess_image
from utils import plot_histogram
import ttkbootstrap as tb

# Initialize GUI
class IrisRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Recognition System")
        self.root.geometry("1200x800")
        self.style = tb.Style("darkly")

        # Header section
        self.header = tb.Frame(root, bootstyle="dark", padding=10)
        self.header.pack(fill="x")

        self.title_label = tb.Label(
            self.header,
            text="Iris Recognition System",
            font=("Segoe UI", 28, "bold"),
            anchor="center",
            bootstyle="inverse-dark",
        )
        self.title_label.pack()

        # Main content section
        self.main_frame = tb.Frame(root, bootstyle="dark", padding=20)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.left_panel = tb.Frame(self.main_frame, bootstyle="secondary", padding=20)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=10)

        self.right_panel = tb.Frame(self.main_frame, bootstyle="secondary", padding=20)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10)

        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Left panel buttons
        self.train_button = tb.Button(
            self.left_panel, text="Train Model", command=self.train_model, bootstyle="success",
            padding=(10, 10)
        )
        self.train_button.pack(fill="x", pady=15)

        self.test_button = tb.Button(
            self.left_panel, text="Test Image", command=self.test_image, bootstyle="primary",
            padding=(10, 10)
        )
        self.test_button.pack(fill="x", pady=15)

        self.view_histogram_button = tb.Button(
            self.left_panel, text="View Histogram", command=self.view_histogram, bootstyle="warning",
            padding=(10, 10)
        )
        self.view_histogram_button.pack(fill="x", pady=15)

        # Image display in right panel
        self.image_frame = tb.LabelFrame(
            self.right_panel, text="Loaded Image", bootstyle="dark", padding=15
        )
        self.image_frame.pack(fill="both", expand=True, pady=10)

        self.image_label = tb.Label(
            self.image_frame, text="No Image Loaded", font=("Segoe UI", 14), bootstyle="secondary"
        )
        self.image_label.pack(anchor="center", expand=True)

        # Results display in right panel
        self.result_frame = tb.LabelFrame(
            self.right_panel, text="Recognition Result", bootstyle="dark", padding=15
        )
        self.result_frame.pack(fill="x", pady=10)

        self.result_label = tb.Label(
            self.result_frame, text="", font=("Segoe UI", 16), bootstyle="white"
        )
        self.result_label.pack()

    def train_model(self):
        dataset_path = filedialog.askdirectory(title="Select Dataset Folder")
        if not dataset_path:
            messagebox.showwarning("Warning", "Dataset folder not selected!")
            return

        compute_average_histograms(dataset_path)
        messagebox.showinfo("Success", "Model training completed! Histograms saved.")

    def test_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Test Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not image_path:
            messagebox.showwarning("Warning", "Test image not selected!")
            return

        self.display_image(image_path)

        histograms_file = "average_histograms.pkl"
        if not os.path.exists(histograms_file):
            messagebox.showerror("Error", "Model not trained yet! Please train the model first.")
            return

        person, similarity = recognize_person(image_path, histograms_file)
        self.result_label.config(text=f"Recognized as: {person}\nSimilarity: {similarity:.2f}")

        # Automatically display histogram
        histogram = preprocess_image(image_path)
        plot_histogram(image_path, title="RGB Histogram")

    def view_histogram(self):
        image_path = filedialog.askopenfilename(
            title="Select Image to View Histogram", filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not image_path:
            messagebox.showwarning("Warning", "Image not selected!")
            return

        histogram = preprocess_image(image_path)
        plot_histogram(image_path, title="RGB Histogram")

    def display_image(self, image_path):
        # Load and display image
        img = Image.open(image_path)
        img = img.resize((400, 400))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk

# Run the GUI
if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = IrisRecognitionApp(root)
    root.mainloop()
