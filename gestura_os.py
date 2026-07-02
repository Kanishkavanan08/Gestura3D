import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ==========================================
# 1. PROCEDURAL 3D GEOMETRY
# ==========================================
def generate_3d_polygon(sides, is_pyramid=False):
    vertices, edges = [], []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        vertices.append((math.cos(angle), math.sin(angle), -1))
    
    if is_pyramid:
        vertices.append((0, 0, 1))
        top_idx = sides
        for i in range(sides):
            edges.append((i, (i+1)%sides))
            edges.append((i, top_idx))
    else:
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            vertices.append((math.cos(angle), math.sin(angle), 1))
        for i in range(sides):
            edges.append((i, (i+1)%sides))
            edges.append((i+sides, (i+1)%sides + sides))
            edges.append((i, i+sides))
            
    return {'vertices': tuple(vertices), 'edges': tuple(edges)}

SHAPES = {'Cube (4-Prism)': generate_3d_polygon(4, False)}
for i in range(3, 10):
    if i != 4: SHAPES[f'{i}-Sided Prism'] = generate_3d_polygon(i, False)
    SHAPES[f'{i}-Sided Pyramid'] = generate_3d_polygon(i, True)
SHAPE_NAMES = list(SHAPES.keys())

class SpatialShape:
    def __init__(self, shape_type, start_x, start_y):
        self.type, self.x, self.y, self.z = shape_type, start_x, start_y, -5.0
        self.scale_x, self.scale_y, self.scale_z = 0.1, 0.1, 0.1
        self.rot_x, self.rot_y = 0, 0
        self.color = (0, 1, 0)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glScalef(self.scale_x, self.scale_y, self.scale_z)
        glBegin(GL_LINES)
        for edge in SHAPES[self.type]['edges']:
            for vertex in edge:
                glColor3fv(self.color)
                glVertex3fv(SHAPES[self.type]['vertices'][vertex])
        glEnd()
        glPopMatrix()

# ==========================================
# 2. GESTURE ALGORITHM ANALYSIS
# ==========================================
def analyze_hand_state(landmarks, w, h):
    # Determine which fingers are extended based on joint vectors
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18] # Lower joints
    fingers_up = []
    
    # Thumb (X-axis comparison)
    if landmarks[tips[0]].x > landmarks[pips[0]].x:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
        
    # Other 4 fingers (Y-axis comparison)
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[pips[i]].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
    # State Classification
    if fingers_up == [0, 1, 0, 0, 0] or fingers_up == [1, 1, 0, 0, 0]:
        state = "POINTING"
    elif fingers_up == [1, 1, 1, 1, 1] or sum(fingers_up) >= 4:
        state = "OPEN_PALM"
    else:
        state = "NEUTRAL"
        
    # Pinch Override (Euclidean distance between Index and Thumb)
    ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
    tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)
    if np.sqrt((ix - tx)**2 + (iy - ty)**2) < 40:
        state = "PINCHING"
        
    return state, ix, iy

# ==========================================
# 3. MASTER OS ENGINE
# ==========================================
def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Gestura3D - 3D Render Window')
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)

    # OS States
    app_mode = "2D_CANVAS" # Tabs: "2D_CANVAS", "3D_BUILDER"
    
    # 2D Canvas Data
    canvas_layer = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_color = (0, 255, 255) # Default Yellow
    colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255)] # R, G, Y, P
    prev_draw_x, prev_draw_y = 0, 0

    # 3D Builder Data
    drawn_objects = []
    active_shape = None
    build_mode = "DRAW"
    selected_idx = 0
    is_pinching = False
    prev_x, prev_y = 0, 0
    start_time = time.time()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        print("\n--- GESTURA OS BOOTED ---")
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            success, frame = cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, int((time.time() - start_time) * 1000))
            
            # --- RENDER TOP NAVIGATION BAR ---
            cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(frame, "2D AIR-CANVAS", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255) if app_mode=="2D_CANVAS" else (100,100,100), 2)
            cv2.putText(frame, "3D SPATIAL BUILDER", (300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255) if app_mode=="3D_BUILDER" else (100,100,100), 2)

            if result.hand_landmarks:
                for hl in result.hand_landmarks:
                    gesture, ix, iy = analyze_hand_state(hl, w, h)
                    gl_x, gl_y = (ix - w/2) / (w/10), -(iy - h/2) / (h/10)
                    
                    # Draw UI Cursor
                    cv2.circle(frame, (ix, iy), 8, draw_color if app_mode == "2D_CANVAS" else (255,255,255), -1)

                    # --- OS TAB SWITCHING (PINCH TO CLICK TABS) ---
                    if gesture == "PINCHING" and iy < 50:
                        if ix < 250: app_mode = "2D_CANVAS"
                        elif ix > 280: app_mode = "3D_BUILDER"
                        continue # Skip other interactions if clicking menu

                    # ==========================================
                    # APP 1: 2D AIR-CANVAS LOGIC
                    # ==========================================
                    if app_mode == "2D_CANVAS":
                        # Draw Color Palette
                        for i, col in enumerate(colors):
                            cv2.rectangle(frame, (20, 100 + i*60), (70, 150 + i*60), col, -1)
                            # Color Select
                            if gesture == "POINTING" and 20 < ix < 70 and (100 + i*60) < iy < (150 + i*60):
                                draw_color = col

                        if gesture == "POINTING":
                            if prev_draw_x != 0:
                                cv2.line(canvas_layer, (prev_draw_x, prev_draw_y), (ix, iy), draw_color, 5)
                            prev_draw_x, prev_draw_y = ix, iy
                            cv2.putText(frame, "WRITING", (ix+20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        else:
                            prev_draw_x, prev_draw_y = 0, 0

                        if gesture == "OPEN_PALM":
                            # Use whole hand radius to erase
                            cv2.circle(canvas_layer, (ix, iy), 60, (0, 0, 0), -1)
                            cv2.circle(frame, (ix, iy), 60, (255, 255, 255), 2) # Show eraser bounds
                            cv2.putText(frame, "ERASING", (ix+70, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                    # ==========================================
                    # APP 2: 3D SPATIAL BUILDER LOGIC
                    # ==========================================
                    elif app_mode == "3D_BUILDER":
                        # Render 3D HUD
                        cv2.rectangle(frame, (20, 70), (150, 110), (0, 140, 255) if build_mode=="DRAW" else (100,100,100), 2)
                        cv2.putText(frame, build_mode, (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        
                        cv2.rectangle(frame, (160, 70), (450, 110), (100,100,100), 2)
                        cv2.putText(frame, f"SHAPE: {SHAPE_NAMES[selected_idx]}", (170, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        if gesture == "PINCHING" and not is_pinching:
                            is_pinching = True
                            
                            # HUD Interactions
                            if 20 < ix < 150 and 70 < iy < 110:
                                build_mode = "MANIPULATE" if build_mode == "DRAW" else "DRAW"
                            elif 160 < ix < 450 and 70 < iy < 110:
                                selected_idx = (selected_idx + 1) % len(SHAPE_NAMES) # Cycle shapes
                            else:
                                # Start Drawing/Grabbing 3D
                                if build_mode == "DRAW":
                                    active_shape = SpatialShape(SHAPE_NAMES[selected_idx], gl_x, gl_y)
                                    active_shape.color = (0, 1, 1)
                                    drawn_objects.append(active_shape)
                                elif build_mode == "MANIPULATE" and drawn_objects:
                                    active_shape = drawn_objects[-1]
                                    active_shape.color = (0, 0, 1)

                        elif gesture == "PINCHING" and is_pinching and active_shape:
                            dx, dy = gl_x - prev_x, gl_y - prev_y
                            if build_mode == "DRAW":
                                active_shape.scale_x = max(0.05, active_shape.scale_x + (dx * 0.5))
                                active_shape.scale_y = max(0.05, active_shape.scale_y + (dy * 0.5))
                                active_shape.scale_z = active_shape.scale_x
                            elif build_mode == "MANIPULATE":
                                active_shape.x += dx
                                active_shape.y += dy
                                active_shape.rot_x += dy * 50
                                active_shape.rot_y += dx * 50

                        elif gesture != "PINCHING" and is_pinching:
                            is_pinching = False
                            if active_shape:
                                active_shape.color = (0, 1, 0)
                                active_shape = None
                        
                        prev_x, prev_y = gl_x, gl_y

            # Composite the 2D canvas onto the camera feed
            if app_mode == "2D_CANVAS":
                # Convert black pixels to transparent, apply drawn lines
                gray_mask = cv2.cvtColor(canvas_layer, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                fg = cv2.bitwise_and(canvas_layer, canvas_layer, mask=mask)
                frame = cv2.add(bg, fg)

            cv2.imshow('Gestura3D - AR HUD', frame)
            
            # Render OpenGL only if we are in 3D Mode
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if app_mode == "3D_BUILDER":
                for obj in drawn_objects: obj.draw()
            pygame.display.flip()
            pygame.time.wait(10)

    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()