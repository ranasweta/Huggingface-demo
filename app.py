import gradio as gr
import torch
import cv2
import pytesseract
import re
import numpy as np

# --- Configuration ---
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.25
print("Model loaded successfully.")

# --- Helper Functions ---
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text).strip()

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 19 , 9
    )
    return adaptive_thresh

# --- Main Processing Function for Gradio ---
def recognize_license_plate(input_image):
    if input_image is None:
        return None, "Please upload an image."

    # Tesseract is pre-installed on Hugging Face Spaces, but we tell pytesseract where it is.
    pytesseract.pytesseract.tesseract_cmd = r'tesseract'

    # Perform detection on the input image (which is RGB from Gradio)
    results = model(input_image)
    
    # Render the results, which gives us an image with boxes drawn on it (in BGR format)
    # results.render() returns a list of images, we take the first one.
    image_with_boxes_bgr = results.render()[0]
    
    # Convert this BGR image back to RGB for correct display in Gradio
    processed_image_rgb = cv2.cvtColor(image_with_boxes_bgr, cv2.COLOR_BGR2RGB)

    # Now, let's get the text
    detections_df = results.pandas().xyxy[0]
    recognized_text = "No license plate detected."

    if len(detections_df) > 0:
        # Use the original RGB input image for cropping to avoid color issues
        row = detections_df.iloc[0]
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Crop from the original input image
        plate_crop = input_image[ymin:ymax, xmin:xmax]

        # Since OpenCV functions expect BGR, convert the crop for preprocessing
        plate_crop_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)

        preprocessed_crop = preprocess_for_ocr(plate_crop_bgr)
        
        custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text_raw = pytesseract.image_to_string(preprocessed_crop, config=custom_config)
        plate_text = clean_text(plate_text_raw)

        recognized_text = plate_text if plate_text else "Could not read text from plate."
        
        # Update the text on our final processed image
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