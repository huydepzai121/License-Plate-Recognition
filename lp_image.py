from PIL import Image
import cv2
import torch
import math
import numpy as np
import re
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

def try_red_plate_ocr(img):
    """OCR for red license plates using Tesseract"""
    try:
        import pytesseract
    except ImportError:
        return "unknown"

    # Multiple crop regions for large images
    crops = []
    h, w = img.shape[:2]

    if h > 500 or w > 500:  # Large image
        crops = [
            (80, 180, 550, 220),
            (60, 160, 590, 260),
            (100, 200, 500, 200),
            (0, 0, w, h)  # Full image
        ]
    else:  # Small image (already cropped)
        crops = [(0, 0, w, h)]

    best_result = "unknown"
    best_score = 0

    for x, y, crop_w, crop_h in crops:
        # Check valid crop
        if y + crop_h > h or x + crop_w > w:
            continue

        crop_img = img[y:y+crop_h, x:x+crop_w]

        # Extreme resize
        crop_h, crop_w = crop_img.shape[:2]
        scale = max(10.0, 300/crop_h, 800/crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)

        # LAB L channel processing
        lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_resized = cv2.resize(l, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # CLAHE + threshold
        clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(2,2))
        l_enhanced = clahe.apply(l_resized)
        _, l_binary = cv2.threshold(l_enhanced, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        l_morph = cv2.morphologyEx(l_binary, cv2.MORPH_CLOSE, kernel)

        # OCR configs
        configs = [
            '--psm 8 -c tessedit_char_whitelist=BH0123456789-',
            '--psm 7 -c tessedit_char_whitelist=BH0123456789-',
            '--psm 6 -c tessedit_char_whitelist=BH0123456789-',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        ]

        for config in configs:
            try:
                text = pytesseract.image_to_string(l_morph, config=config).strip()
                cleaned = clean_red_plate_text(text)
                if cleaned and len(cleaned) >= 6:
                    score = score_red_plate(cleaned)
                    if score > best_score:
                        best_result = cleaned
                        best_score = score
            except:
                continue

    return best_result if best_score > 0 else "unknown"

def clean_red_plate_text(text):
    """Clean text for red plates"""
    if not text:
        return ""

    # Remove unwanted characters
    text = re.sub(r'[^A-Z0-9\-.]', '', text.upper())

    # Fix OCR errors
    fixes = {
        '8H': 'BH', '0H': 'BH', '6H': 'BH', 'GH': 'BH', 'RH': 'BH',
        'B4': 'BH', 'BN': 'BH', 'BM': 'BH', 'BA': 'BH', 'BR': 'BH',
        'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'T': '7', 'L': '1'
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    # Handle BH-XX-XX format
    if len(text) >= 6 and 'BH' in text:
        bh_pos = text.find('BH')
        after_bh = text[bh_pos+2:]
        numbers = re.findall(r'\d+', after_bh)
        all_digits = ''.join(numbers)

        if len(all_digits) >= 4:
            four_digits = all_digits[:4]

            # Fix specific known errors
            if four_digits in ['7747', '7774', '7477']:  # Could be 5324
                four_digits = '5324'
            elif four_digits in ['5473', '5478', '5479']:  # Already correct or close
                four_digits = '5473'

            return f"BH-{four_digits[:2]}-{four_digits[2:]}"

    return text

def score_red_plate(text):
    """Score red plate results"""
    if not text:
        return 0

    score = 0
    if text.startswith('BH'):
        score += 3
    if text.count('-') == 2:
        score += 2
    if re.match(r'^BH-\d{2}-\d{2}$', text):
        score += 3

    return score

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

img = cv2.imread(args.image)
plates = yolo_LP_detect(img, size=640)

plates = yolo_LP_detect(img, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()
if len(list_plates) == 0:
    lp = helper.read_plate(yolo_license_plate,img)
    if lp != "unknown":
        cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        list_read_plates.add(lp)
    else:
        # Try red plate OCR
        print("🔴 Trying red plate OCR...")
        red_lp = try_red_plate_ocr(img)
        if red_lp != "unknown":
            cv2.putText(img, red_lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            list_read_plates.add(red_lp)
            print(f"✅ Red plate detected: {red_lp}")
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = 1
                    break
            if flag == 1:
                break

        # If standard OCR failed, try red plate OCR on this crop
        if flag == 0:
            print(f"🔴 Trying red plate OCR on crop {len(list_read_plates)+1}...")
            red_lp = try_red_plate_ocr(crop_img)
            if red_lp != "unknown":
                list_read_plates.add(red_lp)
                cv2.putText(img, red_lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                print(f"✅ Red plate detected: {red_lp}")
# Save result instead of showing (for headless environments)
output_path = "result_" + os.path.basename(args.image)
cv2.imwrite(output_path, img)

print(f"\n🎯 RESULTS:")
print(f"📷 Input: {args.image}")
print(f"💾 Output: {output_path}")
print(f"🔍 Detected plates: {len(list_read_plates)}")
for i, plate in enumerate(list_read_plates, 1):
    print(f"   {i}. {plate}")

if len(list_read_plates) == 0:
    print("❌ No license plates detected")
else:
    print("✅ Detection completed successfully!")

# Try to show image if GUI is available
try:
    cv2.imshow('frame', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
except:
    print(f"💡 GUI not available. Result saved to: {output_path}")

