import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from PIL import Image

# Adjust imports to match your actual modules
from main import encoder  
# If you've saved your trained model to disk, you can load it:
# model = tf.keras.models.load_model("cnn_model.h5")
model = tf.keras.models.load_model("resnet_model.h5")


def preprocess_image(image_path):
    """Load and preprocess the image to fit the CNN input shape."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    # Expand dims to create batch dimension: (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

def predict_image(image_path):
    """Return top 3 predicted subcategories for the image."""
    processed = preprocess_image(image_path)
    predictions = model.predict(processed)[0]  # shape (num_classes,)
    # Get indices of top-3 predictions
    top_3_indices = predictions.argsort()[-3:][::-1]
    # Map indices back to encoder categories
    top_3_subcategories = [encoder.categories_[0][idx] for idx in top_3_indices]
    # Round or format each probability
    top_3_probs = [f"{predictions[idx]*100:.2f}%" for idx in top_3_indices]
    return list(zip(top_3_subcategories, top_3_probs))

def open_file_dialog():
    """Open file dialog to select an image and show top-3 predictions."""
    file_path = filedialog.askopenfilename(
        filetypes=[("PNG Images", "*.png"), ("JPEG Images", "*.jpg;*.jpeg"), ("All Files", "*.*")]
    )
    if file_path:
        results = predict_image(file_path)
        result_text = "\n".join([f"{cat}: {prob}" for cat, prob in results])
        label_result.config(text=result_text)

# Basic Tkinter GUI
root = tk.Tk()
root.title("Image Classifier")

btn_open = tk.Button(root, text="Select Image", command=open_file_dialog)
btn_open.pack(pady=10)

label_result = tk.Label(root, text="", wraplength=300, justify="left")
label_result.pack(pady=10)

root.mainloop()