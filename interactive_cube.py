import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ==========================================
# 1. 3D CUBE GEOMETRY SETUP
# ==========================================
vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges = (
    (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
    (6,3), (6,4), (6,7), (5,1), (5,4), (5,7)
)
colors = (
    (0, 1, 0), (0, 0, 1), (0, 1, 0), (0, 1, 0),
    (1, 1, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1)
)

def draw_cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(colors[vertex])
            glVertex3fv(vertices[vertex])
    glEnd()

# ==========================================
# 2. TRACKING ENGINE SETUP
# ==========================================
PINCH_THRESHOLD = 45
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# ==========================================
# 3. MAIN APPLICATION LOOP
# ==========================================
def main():
    # Initialize Pygame and OpenGL Window
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Gestura3D - Holographic Workspace')
    
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5) # Push the cube back so we can see it

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    # State tracking for rotation
    prev_x, prev_y = 0, 0
    
    with HandLandmarker.create_from_options(options) as landmarker:
        print("\n--- 3D Workspace Active ---")
        print("1. Pinch your index and thumb together to GRAB.")
        print("2. Move your hand while pinching to SPIN the cube.")
        print("3. Press 'ESC' on the Pygame window to exit.\n")
        
        running = True
        while running:
            # Handle Pygame exit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            # Capture and process webcam feed
            success, frame = cap.read()
            if not success:
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Rotation Logic
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    ix, iy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                    tx, ty = int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h)
                    iz, tz = -hand_landmarks[8].z * 100, -hand_landmarks[4].z * 100
                    
                    # Calculate distance for grab state
                    distance = np.sqrt((ix - tx)**2 + (iy - ty)**2 + (iz - tz)**2)
                    
                    if distance < PINCH_THRESHOLD:
                        # Calculate how much the hand moved since the last frame
                        dx = ix - prev_x
                        dy = iy - prev_y
                        
                        # Apply rotation matrix to the OpenGL space
                        # We divide by 5 to make the rotation speed feel natural
                        glRotatef(np.sqrt(dx**2 + dy**2) / 5, dy, dx, 0)
                    
                    # Update previous coordinates for the next frame
                    prev_x, prev_y = ix, iy

            # Render the OpenGL Frame
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_cube()
            pygame.display.flip()
            pygame.time.wait(10)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()