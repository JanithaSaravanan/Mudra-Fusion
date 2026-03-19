from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
from fpdf import FPDF
from tensorflow.lite.python.interpreter import Interpreter
from story_engine import run_story_engine

app = Flask(__name__)

# --- LOAD ASSETS ---
interpreter = Interpreter(model_path="mudra_mlp_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ["Alapadmam","Anjali","Aralam","Ardhachandran","Ardhapathaka","Berunda","Bramaram","Chakra","Chandrakala","Chaturam","Garuda","Hamsapaksha","Hamsasyam","Kangulam","Kapith","Kapotham","Karkatta","Kartariswastika","Katakamukha_1","Katakamukha_2","Katakamukha_3","Katakavardhana","Katrimukha","Khatva","Kilaka","Kurma","Matsya","Mayura","Mrigasirsha","Mukulam","Mushti","Nagabandha","Padmakosha","Pasha","Pathaka","Pushpaputa","Sakata","Samputa","Sarpasirsha","Shanka","Shivalinga","Shukatundam","Sikharam","Simhamukham","Suchi","Tamarachudam","Tripathaka","Trishulam","Varaha"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- GLOBAL STATE ---
mudra_sequence = []
current_mudra = None
start_time = 0
camera_active = False 
cap = None            
current_story_text = "" # Holds the last generated story for the pipeline

def extract_coords(landmarks):
    wrist = landmarks.landmark[0]
    scale = math.dist([wrist.x, wrist.y], [landmarks.landmark[9].x, landmarks.landmark[9].y]) or 1e-6
    data = []
    for lm in landmarks.landmark:
        data.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, lm.z/scale])
    return data

def get_frames():
    global current_mudra, start_time, mudra_sequence, camera_active, cap
    
    while camera_active:
        if cap is None or not cap.isOpened():
            break

        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        detected_name = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=5),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

            h1 = extract_coords(result.multi_hand_landmarks[0])
            h2 = extract_coords(result.multi_hand_landmarks[1]) if len(result.multi_hand_landmarks) > 1 else [0]*63
            
            X = np.array(h1 + h2, np.float32).reshape(1, -1)
            interpreter.set_tensor(input_details[0]['index'], X)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            
            if np.max(out) > 0.8:
                detected_name = LABELS[np.argmax(out)]
                cv2.putText(frame, f"{detected_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        now = time.time()
        if detected_name:
            if current_mudra != detected_name:
                current_mudra = detected_name
                start_time = now
            elif now - start_time >= 1.0:
                if not mudra_sequence or mudra_sequence[-1] != current_mudra:
                    mudra_sequence.append(current_mudra)
                current_mudra = None

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- CORE ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    global camera_active, cap
    if not camera_active:
        cap = cv2.VideoCapture(0)
        camera_active = True
    return jsonify({"status": "camera started"})

@app.route('/stop_camera')
def stop_camera():
    global camera_active, cap
    camera_active = False
    if cap:
        cap.release()
        cap = None
    return jsonify({"status": "camera stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sequence')
def get_sequence():
    return jsonify(mudra_sequence)

@app.route('/clear')
def clear():
    global mudra_sequence, current_story_text
    mudra_sequence = []
    current_story_text = ""
    return jsonify({"status": "cleared"})

# --- PIPELINE ROUTES ---

@app.route('/generate_story')
def generate_story():
    global mudra_sequence, current_story_text
    if not mudra_sequence:
        return jsonify({"story": "❌ No mudras detected in the sequence yet."})
    
    current_story_text = run_story_engine(mudra_sequence)
    return jsonify({"story": current_story_text})

@app.route('/output_pipeline')
def output_pipeline():
    return render_template('pipeline_choice.html')

@app.route('/pdf_format')
def pdf_format():
    global current_story_text

    pdf = FPDF()
    pdf.add_page()

    # ---------- TITLE ----------
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(0, 102, 204)   # Blue
    pdf.cell(0, 12, "Mudra Story Interpretation", ln=True, align="C")

    pdf.ln(10)

    # ---------- CLEAN TEXT ----------
    clean_text = current_story_text.encode('latin-1', 'ignore').decode('latin-1')

    # ---------- FORMAT STORY ----------
    lines = clean_text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            pdf.ln(4)
            continue

        # Headings
        if line.startswith("🔹") or line.startswith("📜") or line.startswith("🔤") or line.startswith("📘") or line.startswith("🧠") or line.startswith("🎭") or line.startswith("✨"):
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 8, line)

        # Sub headings like Source, Speaker
        elif line.startswith("Source") or line.startswith("Speaker"):
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 8, line)

        # Normal text
        else:
            pdf.set_font("Arial", "", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 8, line)

    # ---------- CREATE STATIC FOLDER ----------
    if not os.path.exists('static'):
        os.makedirs('static')

    pdf_path = "static/output_story.pdf"
    pdf.output(pdf_path)

    return render_template('pdf_view.html', pdf_file="output_story.pdf")

@app.route('/voice_format')
def voice_format():
    global current_story_text
    return render_template('voice_view.html', story=current_story_text)

@app.route('/download_pdf')
def download_pdf():
    return send_file("static/output_story.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)