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

📝 *To add screenshots:*
1. Create a folder named `assets` in your project root.
2. Save your screenshots as `input.png` and `output.png` inside `assets`.
3. They'll render automatically here once pushed to GitHub.

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
