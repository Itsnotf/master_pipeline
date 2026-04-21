import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import os
import urllib.request

# ==========================================
# Setup MediaPipe Hand Landmarker
# ==========================================

def initialize_hand_landmarker(model_path):
    """
    Initialize MediaPipe Hand Landmarker model.
    Downloads model if not exists.
    """
    if not os.path.exists(model_path):
        print("Downloading hand landmarker model...")
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(model_url, model_path)
        print("[OK] Model downloaded")
    
    BaseOptions = base_options.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
    
    return HandLandmarker.create_from_options(options)

# ==========================================
# PC6 Detection Function
# ==========================================

def detect_pc6(image_path, model_path, output_path="output_pc6.jpg"):
    """
    Detect PC6 (Neiguan) acupuncture point on hand.
    
    Args:
        image_path: Path to hand image
        model_path: Path to MediaPipe model
        output_path: Path to save output image
    
    Returns:
        dict with detection results or None if failed
    """
    # Check image exists
    if not os.path.exists(image_path):
        print("[ERROR] Image not found:", image_path)
        return None
    
    # Initialize landmarker
    landmarker = initialize_hand_landmarker(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Convert to RGB and MediaPipe format
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Detect hand landmarks
    results = landmarker.detect(mp_image)
    output = image.copy()
    
    if not results.hand_landmarks:
        print("[ERROR] No hand detected!")
        cv2.imwrite(output_path, output)
        return None
    
    # ==========================================
    # Calculate PC6 (Midline Anatomical Logic)
    # ==========================================
    
    pc6_data = []
    
    for hand_landmarks_list in results.hand_landmarks:
        # Landmark penting
        wrist = hand_landmarks_list[0]       # Wrist
        middle_mcp = hand_landmarks_list[9]  # Middle finger MCP
        
        # Convert ke pixel
        wrist_px = np.array([int(wrist.x * w), int(wrist.y * h)])
        middle_px = np.array([int(middle_mcp.x * w), int(middle_mcp.y * h)])
        
        # === Vector arah telapak (midline tangan)
        palm_vector = middle_px - wrist_px
        palm_length = np.linalg.norm(palm_vector)
        
        if palm_length == 0:
            print("[WARNING] Palm length zero error")
            continue
        
        palm_direction = palm_vector / palm_length
        
        # === Arah ke forearm (kebalikan telapak)
        forearm_direction = -palm_direction
        
        # === Estimasi 2 cun ≈ 0.6 panjang telapak
        pc6_distance = 0.6 * palm_length
        
        # === Titik PC6 (tanpa lateral shift)
        pc6_point = wrist_px + forearm_direction * pc6_distance
        pc6_point = pc6_point.astype(int)
        
        # ==========================================
        # Draw Debug Points
        # ==========================================
        
        # Wrist (biru)
        cv2.circle(output, tuple(wrist_px), 8, (255, 0, 0), -1)
        cv2.putText(output, "Wrist", tuple(wrist_px + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # PC6 (merah)
        cv2.circle(output, tuple(pc6_point), 10, (0, 0, 255), -1)
        cv2.putText(output, "PC6 (Neiguan)", tuple(pc6_point + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        pc6_data.append({
            'wrist': tuple(wrist_px),
            'pc6_point': tuple(pc6_point),
            'confidence': hand_landmarks_list[0].presence if hasattr(hand_landmarks_list[0], 'presence') else None
        })
    
    # Save result
    cv2.imwrite(output_path, output)
    print("[OK] PC6 detected and saved as", output_path)
    
    return {
        'success': True,
        'pc6_data': pc6_data,
        'output_image': output_path
    }

if __name__ == "__main__":
    # Test code
    model_path = "model/hand_landmarker.task"
    image_path = "hand.jpeg"
    detect_pc6(image_path, model_path)
