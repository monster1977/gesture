import cv2
import mediapipe as mp
import time
import urllib.request
import os
import math

# 1. Автоматическое скачивание модели
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Скачивание модели MediaPipe...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Модель загружена.")

# 2. Импорты
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 3. Настройка детектора
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_distance(p1, p2):
    """Вычисляет расстояние между двумя точками"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_angle(p1, p2, p3):
    """Вычисляет угол в точке p2 между векторами p2->p1 и p2->p3"""
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def detect_fingers(landmarks, handedness_str):
    fingers = []
    debug_info = {}
    
    # ===== БОЛЬШОЙ ПАЛЕЦ - КОМБИНИРОВАННЫЙ МЕТОД =====
    
    # Метод 1: Углы в суставах
    # Угол в MCP суставе (точка 2)
    angle_mcp = calculate_angle(landmarks[0], landmarks[2], landmarks[3])
    # Угол в IP суставе (точка 3)
    angle_ip = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
    
    # Если оба угла большие - палец выпрямлен
    thumb_extended = (angle_mcp > 150) and (angle_ip > 150)
    debug_info['angles'] = f"Углы: MCP={angle_mcp:.0f}°, IP={angle_ip:.0f}°"
    
    # Метод 2: Расстояние от кончика до основания указательного пальца
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]  # Основание указательного пальца
    
    dist_tip_to_index = calculate_distance(thumb_tip, index_mcp)
    dist_ip_to_index = calculate_distance(thumb_ip, index_mcp)
    dist_mcp_to_index = calculate_distance(thumb_mcp, index_mcp)
    
    # Кончик должен быть дальше от основания указательного, чем суставы
    thumb_spread = (dist_tip_to_index > dist_ip_to_index * 1.05)
    debug_info['distances'] = f"Расст: T-I={dist_tip_to_index:.3f}, IP-I={dist_ip_to_index:.3f}"
    
    # Метод 3: Z-координата (глубина)
    # Когда большой палец отставлен, его z-координата отличается
    thumb_tip_z = landmarks[4].z
    other_tips_z = (landmarks[8].z + landmarks[12].z + landmarks[16].z + landmarks[20].z) / 4
    z_difference = abs(thumb_tip_z - other_tips_z)
    
    thumb_z_diff = z_difference > 0.015
    debug_info['z_diff'] = f"Z-разница: {z_difference:.4f}"
    
    # Комбинированное решение: палец поднят, если выполняется минимум 2 из 3 условий
    conditions = [thumb_extended, thumb_spread, thumb_z_diff]
    conditions_met = sum(conditions)
    
    if conditions_met >= 2:
        fingers.append(1)
    else:
        fingers.append(0)
    
    debug_info['conditions'] = f"Условия: {conditions_met}/3 (Угл:{thumb_extended}, Расст:{thumb_spread}, Z:{thumb_z_diff})"
    
    # ===== ОСТАЛЬНЫЕ 4 ПАЛЬЦА =====
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip_id, pip_id in zip(tip_ids, pip_ids):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers, debug_info

def recognize_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]: return "Кулак"
    elif fingers == [1, 1, 1, 1, 1]: return "Открытая ладонь (5 пальцев)"
    elif fingers == [0, 1, 0, 0, 0]: return "Указательный палец"
    elif fingers == [0, 1, 1, 0, 0]: return "Знак V / Ножницы"
    elif fingers == [1, 0, 0, 0, 0]: return "Большой палец вверх"
    elif fingers == [0, 0, 0, 0, 1]: return "Мизинец"
    elif fingers == [1, 1, 0, 0, 0]: return "Большой и указательный"
    elif fingers == [0, 1, 1, 1, 0]: return "Три пальца"
    elif fingers == [0, 1, 1, 1, 1]: return "Четыре пальца"
    elif fingers == [1, 1, 1, 0, 0]: return "Три (большой, указательный, средний)"
    else: return "Неизвестный жест"

# Создание детектора
landmarker = HandLandmarker.create_from_options(options)

# Захват видео
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
frame_timestamp_ms = 0

print("Запуск распознавания жестов (Улучшенная версия)...")
print("Нажмите 'q' для выхода")

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    frame_timestamp_ms += 33
    
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if result.hand_landmarks and result.handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            # Отрисовка точек
            for lm in hand_landmarks:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            hand_label = handedness[0].category_name
            fingers, debug_info = detect_fingers(hand_landmarks, hand_label)
            gesture = recognize_gesture(fingers)
            
            # Вывод основной информации
            y_offset = 60 + idx * 150
            cv2.putText(img, f'{hand_label}: {gesture}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            fingers_text = ' '.join(['↑' if f == 1 else '↓' for f in fingers])
            cv2.putText(img, f'Пальцы: {fingers_text}', (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Отладочная информация для большого пальца
            cv2.putText(img, debug_info['angles'], (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, debug_info['distances'], (10, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, debug_info['z_diff'], (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, debug_info['conditions'], (10, y_offset + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Распознавание жестов', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()