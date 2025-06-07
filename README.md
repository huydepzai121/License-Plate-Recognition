# Vietnamese License Plate Recognition

This repository provides you with a detailed guide on how to training and build a Vietnamese License Plate detection and recognition system. This system can work on multiple types of license plates in Vietnam:

- **1 line plates** (standard civilian plates)
- **2 lines plates** (standard civilian plates)
- **Red plates** (military/government vehicles) - **NEW!** ✨

## Installation

```bash
  git clone https://github.com/Marsmallotr/License-Plate-Recognition.git
  cd License-Plate-Recognition

  # install dependencies using pip
  pip install -r ./requirement.txt

  # install Tesseract OCR for red plate recognition (optional but recommended)
  # Ubuntu/Debian:
  sudo apt-get install tesseract-ocr
  pip install pytesseract

  # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  # macOS: brew install tesseract
```

- **Pretrained model** provided in ./model folder in this repo

- **Download yolov5 (old version) from this link:** [yolov5 - google drive](https://drive.google.com/file/d/1g1u7M4NmWDsMGOppHocgBKjbwtDA-uIu/view?usp=sharing)

- Copy yolov5 folder to project folder

## Run License Plate Recognition

```bash
  # run inference on webcam (15-20fps if there is 1 license plate in scene)
  python webcam.py

  # run inference on image (supports all plate types including red plates)
  python demo.py test_image/3.jpg        # standard plate
  python demo.py test_image/bienso2.png  # red plate (military)
  python demo.py test_image/anhdo.jpg    # red plate (military)

  # legacy command (still works)
  python lp_image.py -i test_image/3.jpg

  # run LP_recognition.ipynb if you want to know how model work in each step
```

## Result

### Standard License Plates
![Demo 1](result/image.jpg)

![Vid](result/video_1.gif)

### Red License Plates (Military/Government) - NEW! ✨

The system now supports red license plates with white text, commonly used for military and government vehicles:

- **bienso2.png**: Successfully recognizes `BH-53-24`
- **anhdo.jpg**: Successfully recognizes `BH-54-73`

**Technical Features for Red Plates:**
- Advanced LAB color space processing
- CLAHE enhancement for better contrast
- Tesseract OCR with specialized configurations
- Morphological operations for noise reduction
- Automatic fallback when YOLO detection fails

## Vietnamese Plate Dataset

This repo uses 2 sets of data for 2 stage of license plate recognition problem:

- [License Plate Detection Dataset](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)
- [Character Detection Dataset](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)

Thanks [Mì Ai](https://www.miai.vn/thu-vien-mi-ai/) and [winter2897](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md) for sharing a part in this dataset.

## Training

**Training code for Yolov5:**

Use code in ./training folder
```bash
  training/Plate_detection.ipynb     #for LP_Detection
  training/Letter_detection.ipynb    #for Letter_detection
```
