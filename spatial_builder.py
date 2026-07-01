import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ==========================================
# 1. INBUILT SHAPE GEOMETRY
# ==========================================
SHAPES = {
    'cube': {
        'vertices': ((1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)),
        'edges': ((0,1), (0,3), (0,4), (2,1), (2,3), (2,7), (6,3), (6,4), (6,7), (5,1), (5,4), (5,7))
    },
    'pyramid': {
        'vertices': ((0, 1, 0), (1, -1, -1), (-1, -1, -1), (-1, -1, 1), (1, -1, 1)),
        'edges': ((0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1))
    }
}

class SpatialShape:
    def __init__(self, shape_type, start_x, start_y):
        self.type = shape_type
        self.x, self.y = start_x, start_y
        self.z = -5.0
        self.scale = 0.1
        self.rot_x, self.rot_y = 0, 0
        self.color = (0, 1, 0)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glScalef(self.scale, self.scale, self.scale)
        glBegin(GL_LINES)
        for edge in SHAPES[self.type]['edges']:
            for vertex in edge:
                glColor3fv(self.color)
                glVertex3fv(SHAPES[self.type]['vertices'][vertex])
        glEnd()
        glPopMatrix()

# ==========================================
# 2. ENGINE SETUP
# ==========================================
PINCH_THRESHOLD = 40
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Gestura3D - 3D Canvas')
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    drawn_objects = []
    active_shape = None
    engine_mode = "DRAW"
    selected_type = "cube"
    is_pinching = False
    prev_x, prev_y = 0, 0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        print("\n--- AR UI ACTIVE ---")
        print("Interact with the buttons on your camera feed!")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            success, frame = cap.read()
            if not success: continue
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # -----------------------------------
            # DRAW AR CONTROL PANEL (OPENCV UI)
            # -----------------------------------
            # Mode Button
            mode_color = (0, 200, 0) if engine_mode == "DRAW" else (200, 0, 0)
            cv2.rectangle(frame, (10, 10), (150, 60), mode_color, -1)
            cv2.putText(frame, engine_mode, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Cube Button
            cv2.rectangle(frame, (160, 10), (260, 60), (50, 50, 50), -1)
            cube_text_color = (255,255,255) if selected_type == 'cube' else (150,150,150)
            cv2.putText(frame, "CUBE", (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cube_text_color, 2)
            
            # Pyramid Button
            cv2.rectangle(frame, (270, 10), (390, 60), (50, 50, 50), -1)
            pyr_text_color = (255,255,255) if selected_type == 'pyramid' else (150,150,150)
            cv2.putText(frame, "PYRAMID", (280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pyr_text_color, 2)
            
            # Clear Button
            cv2.rectangle(frame, (400, 10), (500, 60), (0, 0, 200), -1)
            cv2.putText(frame, "CLEAR", (415, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # -----------------------------------
            # SPATIAL MATH & UI COLLISION
            # -----------------------------------
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, int((time.time() - start_time) * 1000))
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    ix, iy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                    tx, ty = int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h)
                    
                    # Draw cursor on index finger
                    cv2.circle(frame, (ix, iy), 8, (0, 255, 255), -1)
                    
                    gl_x, gl_y = (ix - w/2) / (w/10), -(iy - h/2) / (h/10)
                    current_pinch = np.sqrt((ix - tx)**2 + (iy - ty)**2) < PINCH_THRESHOLD

                    ui_clicked_this_frame = False

                    if current_pinch and not is_pinching:
                        is_pinching = True
                        
                        # 1. CHECK UI BUTTON CLICKS
                        if 10 < ix < 150 and 10 < iy < 60:
                            engine_mode = "MANIPULATE" if engine_mode == "DRAW" else "DRAW"
                            ui_clicked_this_frame = True
                        elif 160 < ix < 260 and 10 < iy < 60:
                            selected_type = 'cube'
                            ui_clicked_this_frame = True
                        elif 270 < ix < 390 and 10 < iy < 60:
                            selected_type = 'pyramid'
                            ui_clicked_this_frame = True
                        elif 400 < ix < 500 and 10 < iy < 60:
                            drawn_objects.clear()
                            ui_clicked_this_frame = True
                            
                        # 2. 3D CANVAS INTERACTION (Only if we didn't click the UI)
                        if not ui_clicked_this_frame:
                            if engine_mode == "DRAW":
                                active_shape = SpatialShape(selected_type, gl_x, gl_y)
                                active_shape.color = (1, 1, 0)
                                drawn_objects.append(active_shape)
                            elif engine_mode == "MANIPULATE" and drawn_objects:
                                active_shape = drawn_objects[-1]
                                active_shape.color = (1, 0, 0)

                    elif current_pinch and is_pinching and active_shape:
                        dx, dy = gl_x - prev_x, gl_y - prev_y
                        if engine_mode == "DRAW":
                            active_shape.scale = max(0.1, active_shape.scale + (np.sqrt(dx**2 + dy**2) if dx > 0 else -np.sqrt(dx**2 + dy**2)))
                        elif engine_mode == "MANIPULATE":
                            active_shape.x += dx
                            active_shape.y += dy
                            active_shape.rot_x += dy * 50
                            active_shape.rot_y += dx * 50

                    elif not current_pinch and is_pinching:
                        is_pinching = False
                        if active_shape:
                            active_shape.color = (0, 1, 0)
                            active_shape = None
                    
                    prev_x, prev_y = gl_x, gl_y

            # Render OpenCV UI Window
            cv2.imshow('Gestura3D - AR Control Panel', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): running = False

            # Render Pygame 3D Window
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for obj in drawn_objects: obj.draw()
            pygame.display.flip()
            pygame.time.wait(10)

    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()