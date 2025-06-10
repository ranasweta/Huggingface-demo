# app.py - Final Version for Windows-Trained Model

import gradio as gr
import torch
import cv2
import pytesseract
import re
import numpy as np

# ==============================================================================
#  THE PATHLIB PATCH FOR WINDOWS-TRAINED MODELS
# ==============================================================================
# This block fixes the "cannot instantiate 'WindowsPath' on your system" error.
# It MUST be placed BEFORE `torch.hub.load`.
import pathlib
import platform
# Temporarily patch pathlib if we are on a Linux system (like Hugging Face)
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
# ==============================================================================


# --- Configuration ---
print("Loading YOLOv5 model...")
# Now, load the model. The patch above will prevent it from crashing.
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.25
print("Model loaded successfully.")


# --- Helper Functions ---
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text).strip()

def preprocess_for_ocr(crop):
    # Convert to BGR as OpenCV functions expect this format
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 7
    )
    return adaptive_thresh

# --- Main Processing Function for Gradio ---
def recognize_license_plate(input_image_rgb):
    if input_image_rgb is None:
        return None, "Please upload an image."

    # Tell pytesseract where the tesseract executable is on the Hugging Face server
    pytesseract.pytesseract.tesseract_cmd = r'tesseract'

    # Perform detection on the RGB image from Gradio
    results = model(input_image_rgb)
    
    # Render() draws boxes and returns a BGR image with boxes
    image_with_boxes_bgr = results.render()[0]
    
    # Convert the final display image back to RGB for Gradio
    processed_image_rgb = cv2.cvtColor(image_with_boxes_bgr, cv2.COLOR_BGR2RGB)

    # Now, get the text from the detected region
    detections_df = results.pandas().xyxy[0]
    recognized_text = "No license plate detected."

    if len(detections_df) > 0:
        row = detections_df.iloc[0]
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Crop from the original input image (which is RGB)
        plate_crop_rgb = input_image_rgb[ymin:ymax, xmin:xmax]
        
        preprocessed_crop = preprocess_for_ocr(plate_crop_rgb)
        
        custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text_raw = pytesseract.image_to_string(preprocessed_crop, config=custom_config)
        plate_text = clean_text(plate_text_raw)
        
        recognized_text = plate_text if plate_text else "Could not read text from plate."
        
        # Update the text on our final processed image (which is already RGB)
        # Note: cv2.putText modifies the image in-place
        cv2.putText(processed_image_rgb, recognized_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return processed_image_rgb, recognized_text

# --- Create the Gradio Interface ---
iface = gr.Interface(
    fn=recognize_license_plate,
    inputs=gr.Image(type="numpy", label="Upload Vehicle Image"),
    outputs=[
        gr.Image(type="numpy", label="Processed Image"),
        gr.Textbox(label="Recognized Text")
    ],
    title="License Plate Recognition with YOLOv5 & Tesseract",
    description="Upload an image of a vehicle to detect and read its license plate. The system uses a custom YOLOv5 model for detection and Tesseract for OCR."
)

# --- Launch the App ---
iface.launch()