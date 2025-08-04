# server.py
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =================================================================
# ✨ [핵심 로직] 색상 + 밝기 하이브리드 라인 검출 함수
# =================================================================
def is_line_present(roi_image):
    """
    색상 또는 밝기 분석을 통해 라인의 존재 여부를 판단하는 함수.
    주황색/갈색 계열의 색상 라인 또는 단순 어두운 라인을 모두 감지합니다.
    """
    # --- 1. 색상 분석 (주황/갈색 계열) ---
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([5, 100, 100])
    upper_bound = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    color_pixel_count = cv2.countNonZero(mask)
    
    # 색상으로 검출된 픽셀이 ROI 면적의 5% 이상이면 라인이 있는 것으로 간주
    COLOR_PIXEL_THRESHOLD = roi_image.shape[0] * roi_image.shape[1] * 0.05
    if color_pixel_count > COLOR_PIXEL_THRESHOLD:
        return True

    # --- 2. 밝기 분석 (흑백/어두운 계열) ---
    # 색상으로 검출되지 않은 경우, 흑백으로 변환하여 밝기 분석을 수행
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    mean_intensity = cv2.mean(gray)[0]
    
    # 밝기 임계값 (이 값보다 어두우면 라인으로 간주)
    INTENSITY_THRESHOLD = 180
    if mean_intensity < INTENSITY_THRESHOLD:
        return True

    # 위 두 조건에 모두 해당하지 않으면 라인이 없는 것으로 최종 판단
    return False

def analyze_kit_from_image(img):
    img_height, img_width, _ = img.shape
    debug_image = img.copy()
    final_results = []
    
    strip_x_ratios = [
        (0.10, 0.20), (0.24, 0.34), (0.38, 0.48),
        (0.52, 0.62), (0.66, 0.76), (0.80, 0.90)
    ]

    for i, (x_start_ratio, x_end_ratio) in enumerate(strip_x_ratios):
        x_start = int(img_width * x_start_ratio)
        x_end = int(img_width * x_end_ratio)
        strip = img[0:img_height, x_start:x_end]
        
        cv2.rectangle(debug_image, (x_start, 0), (x_end, img_height), (0, 255, 0), 2)
        
        strip_height = strip.shape[0]
        
        c_y_start, c_y_end = int(strip_height * 0.20), int(strip_height * 0.50)
        c_line_roi = strip[c_y_start:c_y_end, :]
        cv2.rectangle(debug_image, (x_start, c_y_start), (x_end, c_y_end), (255, 0, 0), 2)

        t_y_start, t_y_end = int(strip_height * 0.50), int(strip_height * 0.80)
        t_line_roi = strip[t_y_start:t_y_end, :]
        cv2.rectangle(debug_image, (x_start, t_y_start), (x_end, t_y_end), (0, 0, 255), 2)

        # ✨ [수정 지점] 새로운 하이브리드 검출 함수 사용
        c_line_present = is_line_present(c_line_roi)
        t_line_present = is_line_present(t_line_roi)
        
        print(f"Strip #{i+1} | C-Line Present: {c_line_present}, T-Line Present: {t_line_present}")

        if not c_line_present:
            final_results.append(0)
        elif t_line_present:
            final_results.append(-1)
        else:
            final_results.append(1)

    _, buffer = cv2.imencode('.jpg', debug_image)
    debug_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return debug_image_base64, final_results


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        image_data = base64.b64decode(request.json['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        debug_image_str, results = analyze_kit_from_image(img)
        
        return jsonify({
            'result': results,
            'debugImage': debug_image_str
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)