# main.py
import cv2
import torch

# --- This part is now working correctly ---
try:
    # Path to your local YOLOv5 repository
    yolov5_path =r'D:\projects\LPR\yolov5'
    # Path to your custom weights
    weights_path = r'D:\projects\LPR\runs\train\exp2\weights\best.pt'
    # Load the model from the local source
    model = torch.hub.load(yolov5_path, 'custom', path=weights_path, source='local')
    model.conf = 0.6  # Set a confidence threshold (e.g., 60%)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
# -------------------------------------------


# --- THIS IS THE PART TO FOCUS ON ---

# Initialize the webcam. 0 is usually the built-in webcam.
cap = cv2.VideoCapture(0) 

# **CRUCIAL CHECK 1: Was the camera opened successfully?**
if not cap.isOpened():
    print("Error: Could not open video stream. Is the webcam connected and not in use by another app?")
    exit()

print("🎥 Press 'q' to quit.")

# --- Inside your main.py while loop ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # --- Run Inference ---
    results = model(frame)

    # =========================================================
    #  THIS IS THE LINE YOU NEED TO ADD
    #  It prints the detection summary directly to your console
    results.print()
    # =========================================================

    # This line renders the bounding boxes onto the image for display
    rendered_frame = results.render()[0] 
    
    # Display the resulting frame in the pop-up window
    cv2.imshow("Live OCR Feed", rendered_frame)

    # Wait for 'q' to be pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("👋 Exiting...")
cap.release()
cv2.destroyAllWindows()