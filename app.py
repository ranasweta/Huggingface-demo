import gradio as gr
import torch
import cv2
import pytesseract
import re
import numpy as np

# --- Configuration ---
# We will load the model once when the app starts
print("Loading YOLOv5 model...")
# Using the official ultralytics/yolov5 repo from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.25  # Set a confidence threshold
print("Model loaded successfully.")

# --- Helper Functions ---
def clean_text(text):
    """Removes unwanted characters from OCR output."""
    return re.sub(r'[^A-Z0-9]', '', text).strip()

def preprocess_for_ocr(crop):
    """Applies a robust preprocessing pipeline for better OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 7
    )
    return adaptive_thresh

# --- Main Processing Function for Gradio ---
def recognize_license_plate(input_image):
    """
    Takes an image, detects the license plate, performs OCR,
    and returns the processed image and the recognized text.
    """
    if input_image is None:
        return None, "Please upload an image."

    # Perform detection
    results = model(input_image)
    detections_df = results.pandas().xyxy[0]

    # Draw on a copy of the image
    processed_image = input_image.copy()
    recognized_text = "No license plate detected."

    if len(detections_df) > 0:
        # Process the first detection
        row = detections_df.iloc[0]
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Crop the plate
        plate_crop = processed_image[ymin:ymax, xmin:xmax]

        # Preprocess for OCR
        preprocessed_crop = preprocess_for_ocr(plate_crop)

        # Run Tesseract OCR
        custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text_raw = pytesseract.image_to_string(preprocessed_crop, config=custom_config)
        plate_text = clean_text(plate_text_raw)

        recognized_text = plate_text if plate_text else "Could not read text."
        
        # Draw bounding box and text on the image
        cv2.rectangle(processed_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(processed_image, recognized_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    return processed_image, recognized_text

# --- Create the Gradio Interface ---
iface = gr.Interface(
    fn=recognize_license_plate,
    inputs=gr.Image(type="numpy", label="Upload Vehicle Image"),
    outputs=[
        gr.Image(type="numpy", label="Processed Image"),
        gr.Textbox(label="Recognized Text")
    ],
    title="License Plate Recognition with YOLOv5 & Tesseract",
    description="Upload an image of a vehicle to detect and read its license plate. The system uses a custom YOLOv5 model for detection and Tesseract for OCR.",
    examples=[["car.jpg"]] # You can add example images
)

# --- Launch the App ---
if __name__ == "__main__":
    # For Hugging Face, Tesseract needs to know its path. It's pre-installed.
    pytesseract.pytesseract.tesseract_cmd = r'tesseract'
    # Create an example image file for the Gradio interface
    cv2.imwrite("car.jpg", np.zeros((100, 100, 3), dtype=np.uint8)) 
    iface.launch()