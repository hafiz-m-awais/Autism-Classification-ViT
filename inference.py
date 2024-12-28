import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import threading

# Load the ViT model and processor
MODEL_PATH = "C:/Users/Awais/Desktop/Task9b/Model"  # Adjust the path
image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# Function to handle image selection and inference
def select_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Resize image for ViT model
            image_tk = ImageTk.PhotoImage(image)

            # Update the displayed image
            panel.config(image=image_tk)
            panel.image = image_tk

            # Run inference in a separate thread to avoid UI freezing
            inference_thread = threading.Thread(target=inference, args=(image,))
            inference_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image. {e}")

# Function for running inference and updating the result
def inference(image):
    # Show loading message and progress bar
    result_label.config(text="Processing...")
    result_label.update_idletasks()
    progress.start()

    # Preprocess the image and perform inference
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class].item()

    # Update result after processing
    class_labels = {0: "Autism", 1: "Non-Autism"}
    result_label.config(text=f"Prediction: {class_labels.get(predicted_class, 'Unknown Class')}\nConfidence: {confidence:.2f}")
    progress.stop()  # Stop progress bar

# Function to clear the image and reset the result
def clear_image():
    panel.config(image='')
    result_label.config(text="Prediction: ")

# Create the main window
root = tk.Tk()
root.title("Autism Classification Inference")
root.geometry("700x600")
root.config(bg="#f4f4f9")  # Light gray background

# Header frame (with dark blue background)
header_frame = tk.Frame(root, bg="#2C3E50", pady=15)
header_frame.pack(fill="x")
header_label = tk.Label(header_frame, text="Autism Classification using Vision Transformer", font=("Helvetica", 18, "bold"), fg="white", bg="#2C3E50")
header_label.pack()

# Image selection frame
image_frame = tk.Frame(root, bg="#f4f4f9", pady=20)
image_frame.pack()

# Image panel for displaying the uploaded image
panel = tk.Label(image_frame, bg="#f4f4f9")
panel.pack()

# Buttons and Result frame
button_frame = tk.Frame(root, bg="#f4f4f9", pady=20)
button_frame.pack()

select_button = tk.Button(button_frame, text="Select Image", command=select_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="raised", width=20)
select_button.grid(row=0, column=0, padx=15)

clear_button = tk.Button(button_frame, text="Clear Image", command=clear_image, font=("Helvetica", 12), bg="#FF5733", fg="white", relief="raised", width=20)
clear_button.grid(row=0, column=1, padx=15)

# Result label to display the prediction
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14), bg="#f4f4f9", fg="#333333")
result_label.pack(pady=20)

# Progress bar (loading indicator)
progress = ttk.Progressbar(root, orient="horizontal", length=200, mode="indeterminate")
progress.pack(pady=10)

# Footer
footer_frame = tk.Frame(root, bg="#2C3E50", pady=15)
footer_frame.pack(fill="x")
footer_label = tk.Label(footer_frame, text="Powered by ViT Model | Autism Classification", font=("Helvetica", 10), fg="white", bg="#2C3E50")
footer_label.pack()

# Run the Tkinter event loop
root.mainloop()
