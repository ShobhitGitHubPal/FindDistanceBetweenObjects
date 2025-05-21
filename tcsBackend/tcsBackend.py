
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from flask import Flask, Response
from flask_socketio import SocketIO, emit
import threading
import time
from playsound import playsound
import pygame
import threading

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load YOLO model
MODEL_PATH = "C:/Users/com/Desktop/findDistanceBetweenBothObject/tcsBackend/yolov8n.pt"
model = YOLO(MODEL_PATH)

# Alarm sound file path
ALARM_SOUND = "C:/Users/com/Desktop/findDistanceBetweenBothObject/danger-alarm-23793.mp3"
alarm_playing = False
active_alarm = False
PIXEL_TO_CM_SCALE = 0.26  # You might need to adjust this scale for your camera

# Initialize pygame mixer for sound
pygame.mixer.init()

def play_alarm():
    """Trigger Alarm Sound repeatedly until stop condition is met."""
    global alarm_playing, active_alarm
    if not alarm_playing and not active_alarm:
        alarm_playing = True
        active_alarm = True
        print("Playing alarm...")
        # Play alarm sound using pygame (non-blocking)
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play(loops=0, start=0.0)

def stop_alarm():
    """Stop Alarm."""
    global alarm_playing, active_alarm
    pygame.mixer.music.stop()  # Stop the sound immediately
    alarm_playing = False
    active_alarm = False
    print("Alarm stopped")

def calculate_pixel_distance(p1, p2):
    """Compute Euclidean distance in pixels."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_real_distance(p1, p2):
    """Convert pixel distance to real-world distance (cm)."""
    return calculate_pixel_distance(p1, p2) * PIXEL_TO_CM_SCALE


def process_frame(frame):
    """Perform object detection and process distances."""
    results = model.predict(frame, conf=0.5, show=False)[0]
    detected_bottle = []
    detected_chairs = []
    real_distance = []

    print(f"Detected objects: {results.names}")
    
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        class_name = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        if class_name == "bottle":
            detected_bottle.append(obj_center)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "bottle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        elif class_name == "chair":
            detected_chairs.append(obj_center)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)          
            cv2.circle(frame, obj_center, 150, (0, 0, 255), 2)  


    for person in detected_bottle:
        for chair in detected_chairs:
            real_distance = calculate_real_distance(chair, person)
            print(f"Real Distance: {real_distance} cm")

            if real_distance <= 47:
                print("ALERT TRIGGERED!")
                play_alarm()
                cv2.putText(frame, "ALERT!", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                stop_alarm()

            # Draw a line between the bottle and the chair
            cv2.line(frame, person, chair, (0, 255, 255), 2)
            cv2.putText(frame, f"{real_distance:.2f} cm", (person[0], person[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Convert frame to base64 for sending via SocketIO
    _, buffer = cv2.imencode(".jpg", frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    # Send data to the frontend via SocketIO
    socketio.emit("update", {
        "frame": frame_base64, 
        "bottle_count": len(detected_bottle),
        "chair_count": len(detected_chairs),
        "distance_info": real_distance
    })

    return frame


def generate_frames():
    print("""Capture frames from webcam and process them.""")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process each frame
        frame = process_frame(frame)

        # Encode frame to bytes for sending to the browser
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    print('Starting video stream...')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask and SocketIO server
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)

