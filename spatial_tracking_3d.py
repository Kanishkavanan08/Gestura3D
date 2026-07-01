import cv2
import mediapipe as mp
import time
import numpy as np

# ==========================================
# 1. SETUP ENGINE & SMOOTHING VARIABLES (ROUTE 1)
# ==========================================
# Exponential Moving Average (EMA) smoothing factor (0.0 to 1.0)
# Lower values = smoother but more lag; Higher values = snappier but more jitter
ALPHA = 0.35
smoothed_x, smoothed_y, smoothed_z = 0, 0, 0
is_initialized = False

# ==========================================
# 2. SETUP GESTURE CONFIGURATION (ROUTE 2)
# ==========================================
# Threshold for a "Pinch/Grab" gesture (Euclidean distance in 3D pixels)
PINCH_THRESHOLD = 45 

# Setup the modern Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)
start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    print("\n--- Gestura3D Active: Tracking, Filtering, & Gestures Enabled ---")
    print("Press 'q' in the video window to quit.\n")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Extract raw coordinates
                # ID 8 = Index Tip, ID 4 = Thumb Tip
                index_tip = hand_landmarks[8]
                thumb_tip = hand_landmarks[4]
                
                # Convert normalized coordinates to absolute screen pixels
                raw_ix, raw_iy = int(index_tip.x * w), int(index_tip.y * h)
                raw_iz = round(-index_tip.z * 100, 2)
                
                raw_tx, raw_ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                raw_tz = round(-thumb_tip.z * 100, 2)
                
                # ----------------------------------------------------
                # ROUTE 1: ALGORITHMIC NOISE FILTERING (EMA)
                # ----------------------------------------------------
                if not is_initialized:
                    smoothed_x, smoothed_y, smoothed_z = raw_ix, raw_iy, raw_iz
                    is_initialized = True
                else:
                    smoothed_x = (ALPHA * raw_ix) + ((1 - ALPHA) * smoothed_x)
                    smoothed_y = (ALPHA * raw_iy) + ((1 - ALPHA) * smoothed_y)
                    smoothed_z = (ALPHA * raw_iz) + ((1 - ALPHA) * smoothed_z)
                
                # Cast back to integers for rendering/coordinates
                cx, cy, cz = int(smoothed_x), int(smoothed_y), round(smoothed_z, 2)
                
                # ----------------------------------------------------
                # ROUTE 2: GESTURE STATE MACHINE (PINCH DETECTION)
                # ----------------------------------------------------
                # Calculate the 3D Euclidean distance between thumb and index finger
                distance = np.sqrt((raw_ix - raw_tx)**2 + (raw_iy - raw_ty)**2 + (raw_iz - raw_tz)**2)
                
                # Determine state
                if distance < PINCH_THRESHOLD:
                    gesture_state = "GRABBING"
                    color = (0, 0, 255)  # Red for holding/grabbing
                else:
                    gesture_state = "HOVERING"
                    color = (255, 0, 0)  # Blue for open/hovering
                
                # ----------------------------------------------------
                # ROUTE 3: VISUAL MATRIX DISPLAY
                # ----------------------------------------------------
                # Draw tracking markers on index and thumb
                cv2.circle(frame, (cx, cy), 8, color, cv2.FILLED)
                cv2.circle(frame, (raw_tx, raw_ty), 6, (0, 255, 255), cv2.FILLED)
                # Draw a line between them to visualize interaction distance
                cv2.line(frame, (cx, cy), (raw_tx, raw_ty), (0, 255, 0), 1)
                
                # Overlay real-time telemetry details on screen
                cv2.putText(frame, f"State: {gesture_state}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.putText(frame, f"Spatial Link: X:{cx} Y:{cy} Z:{cz}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Terminal readout
                print(f"[{gesture_state}] Coord Matrix -> X: {cx} | Y: {cy} | Z: {cz} | Dist: {round(distance, 1)}")
        else:
            is_initialized = False  # Reset filter if hand leaves the frame

        cv2.imshow('Gestura3D - Core Engine Workspace', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()