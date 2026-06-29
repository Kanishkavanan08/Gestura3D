import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup webcam capture
cap = cv2.VideoCapture(0)

# Initialize Hand Tracking
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    print("Tracking started. Move your hand closer/further from the webcam to see Z-depth change.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Get the Index Finger Tip Landmark (ID 8)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 2. Pixel coordinates for X and Y
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                
                # 3. Z-depth raw coordinate (relative to wrist)
                cz_raw = index_tip.z
                
                # 4. Map the raw Z data to a cleaner format for 3D engine scales
                # MediaPipe Z values are negative when closer to the camera, positive when further.
                # Let's invert it for standard distance metrics (Higher value = closer)
                cz_mapped = round(-cz_raw * 100, 2)

                # Display the coordinates on screen
                cv2.putText(frame, f"3D Space: X:{cx} Y:{cy} Z:{cz_mapped}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Print coordinates to console for verification
                print(f"X: {cx} | Y: {cy} | Z: {cz_mapped}")

                # Draw the connections
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the output frame
        cv2.imshow('3D Spatial Tracking Canvas', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()