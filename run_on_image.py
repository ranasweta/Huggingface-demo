# run_on_image.py (Version 5 - With Adaptive Thresholding for Motion Blur)

import cv2
import torch
import pytesseract
import re

# --- CONFIGURATION ---
# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Paths to your model and image
YOLOV5_PATH = r'D:\projects\LPR\yolov5'  # Path to your local YOLOv5 repository
WEIGHTS_PATH = r'D:\projects\LPR\runs\train\exp2\weights\best.pt'
IMAGE_PATH = r'D:\projects\LPR\image.png' # <--- MAKE SURE THIS IS CORRECT
CONFIDENCE_THRESHOLD = 0.5

# --- HELPER FUNCTIONS ---

def clean_text(text):
    """Removes spaces and unwanted characters from OCR output."""
    return re.sub(r'[^A-Z0-9]', '', text).strip()

def preprocess_for_ocr(crop):
    """
    Applies a new preprocessing pipeline using Adaptive Thresholding,
    which is highly effective for images with motion blur and uneven lighting.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply a slight blur to reduce high-frequency noise.
    #    This helps the adaptive thresholding algorithm work better.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. *** The Key Step: Adaptive Thresholding ***
    #    We use ADAPTIVE_THRESH_GAUSSIAN_C, which is often superior.
    #    blockSize (e.g., 15) must be an odd number. It's the size of the pixel neighborhood.
    #    C (e.g., 7) is a constant subtracted from the mean. It's for fine-tuning.
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 
        255,  # Max value to assign
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY,  # We want black text on a white background
        19, # blockSize
        9  # C
    )
    
    return adaptive_thresh

# --- MAIN SCRIPT ---

# 1. Load Model
print("Loading model...")
model = torch.hub.load(YOLOV5_PATH, 'custom', path=WEIGHTS_PATH, source='local')
model.conf = CONFIDENCE_THRESHOLD
print("Model loaded successfully.")

# 2. Load Image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not read image at {IMAGE_PATH}")
    exit()

# 3. Perform Detection
print("Performing detection...")
results = model(image)
detections_df = results.pandas().xyxy[0]
print(f"Found {len(detections_df)} potential detections.")

# 4. Process Detections and Perform OCR
for index, row in detections_df.iterrows():
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    confidence = row['confidence']
    label = row['name']
    
    print(f"\nProcessing detection {index+1}: '{label}' with confidence {confidence:.2f}")

    plate_crop = image[ymin:ymax, xmin:xmax]
    preprocessed_crop = preprocess_for_ocr(plate_crop)

    # --- Tesseract ---
    # --psm 7: Treat as a single line of text. Good for this type of plate.
    custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    plate_text_raw = pytesseract.image_to_string(preprocessed_crop, config=custom_config)
    
    plate_text = clean_text(plate_text_raw)
    
    print(f"✅ OCR Result: '{plate_text}'")
    
    # --- Draw Results on Image ---
    # Basic formatting to add spaces back for display
    formatted_text = ""
    if len(plate_text) >= 4:
        formatted_text = f"{plate_text[0:2]} {plate_text[2:4]} {plate_text[4:]}"
    else:
        formatted_text = plate_text
        
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    cv2.putText(image, formatted_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # --- DEBUG: See what Tesseract is seeing ---
    cv2.imshow("Preprocessed for OCR", preprocessed_crop)

# 5. Display Final Result
print("\nDisplaying final result. Press any key to exit.")
cv2.imshow("License Plate Detection and OCR", image)
cv2.waitKey(0)
cv2.destroyAllWindows()