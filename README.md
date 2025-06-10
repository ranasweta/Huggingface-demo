# 📸 Advanced License Plate Recognition with YOLOv5 & Tesseract

This project provides a complete and robust pipeline for detecting and recognizing vehicle license plates from images. It leverages a custom-trained **YOLOv5** model for high-accuracy object detection and the **Tesseract OCR** engine for extracting alphanumeric text, all tied together with a sophisticated image preprocessing pipeline.

---

## ✨ Live Demo on Hugging Face

Experience the power of this model firsthand! I have deployed a user-friendly web interface on Hugging Face Spaces where you can upload your own images and see the detection results instantly.

➡️ **[Click here to try the Live Demo](https://ranasweta-license-plate-recognition-demo.hf.space/?logs=container&__theme=system&deep_link=5XQgD5aHYlg)**  


(![alt text](https://github.com/ranasweta/Huggingface-demo/blob/main/Demo_online.png))


---

## 🌟 Key Features

- 🔍 **High-Accuracy Detection:** A custom-trained YOLOv5 model pinpoints license plates with precision.
- 🚀 **Robust & Deployable:** The entire pipeline is containerized and deployed on Hugging Face, showcasing a real-world MLOps workflow.
- 🧠 **Intelligent Preprocessing:** Smart preprocessing routine (adaptive thresholding, noise reduction) before OCR improves accuracy under challenging conditions.
- 📦 **Clean & Modular Code:** The codebase is well-structured and documented for easy understanding and extensibility.

---

## 🎯 Project Showcase & Outputs

Below is an example of the system identifying and drawing a bounding box around a license plate.

| Input Image | Detection Result |
|-------------|------------------|
| ![Input Image](https://github.com/ranasweta/Huggingface-demo/blob/main/image.png) | ![Output Image](https://github.com/ranasweta/Huggingface-demo/blob/main/output.png) |



---

## 🛠️ Running the Project Locally (with Live Webcam)

While the Hugging Face demo supports image uploads, the full project also runs on live webcam input.

### 1. Prerequisites

- Python 3.8+
- Git
- **Tesseract OCR** (must be installed separately):

```bash
# On Ubuntu/Debian
sudo apt install tesseract-ocr

# On Mac (with Homebrew)
brew install tesseract
```
## 🛠️ Setup & Installation

Follow the steps below to set up the project on your local machine:

---

### 📦 1. Clone the Repository

```bash
git clone https://github.com/ranasweta/License_Detector_Live

cd License_Detector_Live
```

### 🧪 2. Create and Activate a Virtual Environment

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate
```
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt

```
---
## 3. Running the Application
### 3.1. Running the Application with Webcam Input
```bash
python run_on_webcam.py
```
### 3.2. Running the Application with image input
```bash
python run_on_image.py

```
##📜 Dependencies (requirements.txt)
```bash
torch
opencv-python
opencv-python-headless
pytesseract
ultralytics
numpy
gradio
seaborn
pandas
matplotlib
```
---
## 📞 Contact

**Rana Sweta**  
📧 Email: [
    ranasweta2005@gmail.com
](mailto:ranasweta2005@gmail.com
)  
🤗 Hugging Face: [ranasweta
](https://huggingface.co/spaces/ranasweta
)  
🔗 Live Demo: [License Plate Recognition Demo](https://huggingface.co/spaces/ranasweta/License-Plate-Recognition-Demo)



