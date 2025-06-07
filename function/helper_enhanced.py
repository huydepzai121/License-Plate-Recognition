import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_license_plate_text(text):
    """Làm sạch text biển số - TÍNH NĂNG CHÍNH"""
    if not text:
        return ""
    
    # Loại bỏ ký tự không mong muốn
    text = re.sub(r'[^A-Z0-9\-.]', '', text.upper())
    
    # Sửa các lỗi OCR thông thường - TÍNH NĂNG MỚI
    replacements = {
        'O': '0', 'I': '1', 'S': '5', 'Z': '2',
        'B': '8', 'G': '6', 'Q': '0', 'D': '0'
    }
    
    # Chỉ thay thế trong phần số
    parts = text.split('-')
    if len(parts) == 2:
        for old, new in replacements.items():
            if len(parts[1]) > 0:
                parts[1] = parts[1].replace(old, new)
        text = '-'.join(parts)
    
    return text

def preprocess_for_red_plate(img):
    """Xử lý đặc biệt cho biển số đỏ - TÍNH NĂNG MỚI"""
    try:
        import cv2
        import numpy as np
        
        # Chuyển sang HSV để tách màu đỏ
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Định nghĩa range cho màu đỏ
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Tạo mask cho màu đỏ
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Nếu có nhiều pixel đỏ, đây có thể là biển đỏ
        red_ratio = np.sum(red_mask > 0) / (img.shape[0] * img.shape[1])
        
        if red_ratio > 0.3:  # Nếu >30% là màu đỏ
            # Tạo ảnh với text trắng trên nền đen
            result = np.zeros_like(img)
            result[red_mask == 0] = [255, 255, 255]  # Text trắng
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), True
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), False
    except ImportError:
        logger.warning("OpenCV not available")
        return img, False
    except Exception as e:
        logger.error(f"Error in preprocess_for_red_plate: {e}")
        return img, False

def read_plate_multi_ocr(img):
    """Đọc biển số sử dụng nhiều phương pháp OCR - TÍNH NĂNG MỚI"""
    try:
        # Kiểm tra và xử lý biển đỏ
        processed_img, is_red_plate = preprocess_for_red_plate(img)
        
        # Thử EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(processed_img if is_red_plate else img, detail=0, paragraph=False)
            if results:
                text = ''.join(results).strip()
                cleaned = clean_license_plate_text(text)
                if cleaned and len(cleaned) >= 7:
                    logger.info(f"EasyOCR success: {cleaned}")
                    return cleaned
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
        
        # Thử Tesseract
        try:
            import pytesseract
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            text = pytesseract.image_to_string(processed_img if is_red_plate else img, config=config).strip()
            cleaned = clean_license_plate_text(text)
            if cleaned and len(cleaned) >= 7:
                logger.info(f"Tesseract success: {cleaned}")
                return cleaned
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
        
        return "unknown"
    except Exception as e:
        logger.error(f"Error in read_plate_multi_ocr: {e}")
        return "unknown"

# Các function khác từ helper.py gốc
def linear_equation(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf'), x1
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1 if x1 != 0 else 0
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    try:
        a, b = linear_equation(x1, y1, x2, y2)
        if a == float('inf'):
            return abs(x - b) <= 3
        y_pred = a * x + b
        return abs(y_pred - y) <= 3
    except:
        return False

def read_plate(yolo_license_plate, im):
    """Hàm đọc biển số chính - CẢI THIỆN từ phiên bản gốc"""
    try:
        # Thử phương pháp OCR trực tiếp trước - TÍNH NĂNG MỚI
        multi_ocr_result = read_plate_multi_ocr(im)
        if multi_ocr_result != "unknown":
            logger.info(f"Multi-OCR success: {multi_ocr_result}")
            return multi_ocr_result
        
        # Nếu OCR trực tiếp không thành công, dùng YOLO
        LP_type = "1"
        results = yolo_license_plate(im)
        bb_list = results.pandas().xyxy[0].values.tolist()
        
        # Giảm điều kiện strict - TÍNH NĂNG MỚI: cho phép từ 5-12 ký tự
        if len(bb_list) == 0 or len(bb_list) < 5 or len(bb_list) > 12:
            logger.info(f"YOLO detection failed, bb_list length: {len(bb_list)}")
            return multi_ocr_result
        
        center_list = []
        y_sum = 0
        
        for bb in bb_list:
            x_c = (bb[0] + bb[2]) / 2
            y_c = (bb[1] + bb[3]) / 2
            y_sum += y_c
            center_list.append([x_c, y_c, bb[-1]])
        
        # Phân loại biển 1 dòng hay 2 dòng
        if len(center_list) >= 2:
            l_point = min(center_list, key=lambda x: x[0])
            r_point = max(center_list, key=lambda x: x[0])
            
            for ct in center_list:
                if l_point[0] != r_point[0]:
                    if not check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                        LP_type = "2"
                        break
        
        y_mean = int(y_sum / len(bb_list))
        
        # Xây dựng chuỗi biển số
        line_1 = []
        line_2 = []
        license_plate = ""
        
        if LP_type == "2":
            for c in center_list:
                if int(c[1]) > y_mean:
                    line_2.append(c)
                else:
                    line_1.append(c)
            
            for l1 in sorted(line_1, key=lambda x: x[0]):
                license_plate += str(l1[2])
            license_plate += "-"
            for l2 in sorted(line_2, key=lambda x: x[0]):
                license_plate += str(l2[2])
        else:
            for l in sorted(center_list, key=lambda x: x[0]):
                license_plate += str(l[2])
        
        # Làm sạch kết quả - TÍNH NĂNG MỚI
        license_plate = clean_license_plate_text(license_plate)
        
        if len(license_plate) >= 7:
            logger.info(f"YOLO success: {license_plate}")
            return license_plate
        else:
            logger.info(f"YOLO result too short: {license_plate}")
            return multi_ocr_result
            
    except Exception as e:
        logger.error(f"Error in read_plate: {e}")
        return "unknown"
