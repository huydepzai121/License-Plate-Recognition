from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
import os
import time
import argparse
import function.helper as helper

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

print("Đang tải models...")
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

print("Đang xử lý ảnh...")
img = cv2.imread(args.image)
if img is None:
    print(f"Không thể đọc ảnh: {args.image}")
    exit(1)

print(f"Kích thước ảnh: {img.shape}")

plates = yolo_LP_detect(img, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()

print(f"Tìm thấy {len(list_plates)} biển số")

if len(list_plates) == 0:
    print("Không tìm thấy biển số, thử đọc toàn bộ ảnh...")
    lp = helper.read_plate(yolo_license_plate, img)
    if lp != "unknown":
        print(f"Đọc được biển số: {lp}")
        list_read_plates.add(lp)
    else:
        print("Không đọc được biển số")
else:
    for i, plate in enumerate(list_plates):
        print(f"Xử lý biển số {i+1}...")
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = img[y:y+h, x:x+w]
        
        # Lưu ảnh crop để debug
        crop_filename = f"crop_{i}.jpg"
        cv2.imwrite(crop_filename, crop_img)
        print(f"Đã lưu ảnh crop: {crop_filename}")
        
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    print(f"Đọc được biển số: {lp}")
                    flag = 1
                    break
            if flag == 1:
                break
        
        if flag == 0:
            print(f"Không đọc được biển số từ vùng {i+1}")

# Lưu kết quả vào ảnh output
output_img = img.copy()
for i, plate in enumerate(list_plates):
    x1, y1, x2, y2 = int(plate[0]), int(plate[1]), int(plate[2]), int(plate[3])
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color=(0, 0, 225), thickness=2)

# Hiển thị kết quả đọc được
if list_read_plates:
    print(f"\nKết quả cuối cùng:")
    for i, lp in enumerate(list_read_plates):
        print(f"  Biển số {i+1}: {lp}")
        # Thêm text vào ảnh
        cv2.putText(output_img, lp, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
else:
    print("\nKhông đọc được biển số nào!")

# Lưu ảnh kết quả
output_filename = "result_output.jpg"
cv2.imwrite(output_filename, output_img)
print(f"\nĐã lưu kết quả vào: {output_filename}")
print("Setup hoàn tất!")
