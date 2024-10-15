from flask import Flask, request, Response, render_template, redirect, url_for, jsonify, send_file
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import pandas as pd
from flask import send_file
from threading import Lock

lock = Lock()

app = Flask(__name__)
CORS(app)

cameras = []
registered_names = []
detected_faces = []
detection_interval = 10  # Default 10 seconds

DETECTED_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detected_faces')
EXCEL_FILE_PATH = "detected_faces.xlsx"

lock = Lock()

# Ensure the 'detected_faces' directory exists
if not os.path.exists(DETECTED_FACES_DIR):
    os.makedirs(DETECTED_FACES_DIR)

alarm_config = {
    'enabled': False,
    'target': '',
    'email': ''
}

try:
    with open('alarm_config.json', 'r') as f:
        alarm_config = json.load(f)
except FileNotFoundError:
    pass

# Load registered faces
try:
    data = pd.read_csv('registered_faces.csv')
    known_face_names = data['name'].tolist()
    known_face_encodings = [np.array(eval(encoding)) for encoding in data['encoding']]
except FileNotFoundError:
    known_face_names = []
    known_face_encodings = []

def register_face(name, camera_id):
    global known_face_names, known_face_encodings
    
    video_capture = cv2.VideoCapture(camera_id)
    if not video_capture.isOpened():
        return "Error: Cannot open camera."

    face_encodings_list = []
    samples_collected = 0

    while samples_collected < 5:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            face_encodings_list.append(face_encodings[0])
            samples_collected += 1

    video_capture.release()

    if samples_collected == 5:
        average_encoding = np.mean(face_encodings_list, axis=0)
        if name not in known_face_names:  # Eliminate duplicates
            known_face_encodings.append(average_encoding)
            known_face_names.append(name)

            data = pd.DataFrame({'name': known_face_names, 'encoding': [list(enc) for enc in known_face_encodings]})
            data.to_csv('registered_faces.csv', index=False)
        return f"Face registered successfully for {name}"
    else:
        return "Failed to collect enough face samples."

def gen_frames(camera_id):
    video_capture = cv2.VideoCapture(camera_id)
    last_detection_time = {}
    
    if not video_capture.isOpened():
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n'
        return

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    confidence = int((1 - face_distances[first_match_index]) * 100)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                current_time = datetime.now()
                if name not in last_detection_time or (current_time - last_detection_time[name]).total_seconds() > detection_interval:
                    last_detection_time[name] = current_time

                    # Save detected face image
                    face_image = frame[top:bottom, left:right]
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{name}_{timestamp}.jpg"
                    image_path = os.path.join(DETECTED_FACES_DIR, image_filename)
                    cv2.imwrite(image_path, face_image)

                    # Prepare data for Excel
                    data_entry = {
                        'name': name,
                        'confidence': confidence,
                        'camera': f"Camera {camera_id}",
                        'time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'image': image_filename  # Use just the filename for the Excel entry
                    }
                    # Save the entry to Excel
                    with lock:  # Use lock to prevent race conditions
                        save_to_excel([data_entry])  # Save each entry immediately
                        detected_faces.append(data_entry)

                    #print(f"Added {name} to detected_faces with confidence {confidence}.")
                    #print("inner loop", data_entry)  # Check updated entry

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            
# Function to save detected faces to Excel
def save_to_excel(data):
    if os.path.exists(EXCEL_FILE_PATH):
        # Load existing data
        existing_data = pd.read_excel(EXCEL_FILE_PATH)
        # Append new data
        updated_data = pd.concat([existing_data, pd.DataFrame(data)], ignore_index=True)
    else:
        # Create a new DataFrame
        updated_data = pd.DataFrame(data)

    # Save the updated DataFrame back to Excel
    updated_data.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')


def clear_excel_file():
    if os.path.exists(EXCEL_FILE_PATH):
        os.remove(EXCEL_FILE_PATH)  # Delete the existing file to start fresh for the session


def send_alert_email(name, confidence, camera):
    sender_email = "ramkumar@gemicates.in"  # Replace with your email
    sender_password = "Munivalli143$"  # Replace with your password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = alarm_config['email']
    msg['Subject'] = f"Alert: {alarm_config['target']} Detected"
    
    body = f"{name} detected with {confidence}% confidence on {camera}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent to {alarm_config['email']} successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

            
@app.route('/')
def login():
    global detected_faces
    detected_faces = []  # Clear the detected faces for the new session
    return render_template('login.html')

@app.route('/index.html')
def index():
    return render_template('index.html',cameras=cameras, registered_names=list(set(known_face_names)))

@app.route('/add_camera', methods=['POST'])
def add_camera():
    camera_id = request.form['camera_id']
    cameras.append(camera_id)
    return redirect(url_for('index'))

@app.route('/register_face', methods=['POST'])
def register_face_route():
    name = request.form['name']
    camera_id = request.form['camera_id']
    result = register_face(name, int(camera_id))
    return jsonify({'message': result})

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(int(camera_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_faces')
def get_detected_faces():
    global detected_faces
    faces = detected_faces
    detected_faces = []  # Clear the list after sending
    return jsonify(faces)

@app.route('/set_detection_interval', methods=['POST'])
def set_detection_interval():
    global detection_interval
    interval = request.form.get('interval', type=int)
    unit = request.form.get('unit')
    if unit == 'minutes':
        interval *= 60
    detection_interval = interval
    return jsonify({'message': f'Detection interval set to {interval} seconds'})

@app.route('/detected_face_image/<path:filename>')
def detected_face_image(filename):
    return send_file(os.path.join(DETECTED_FACES_DIR, filename), mimetype='image/jpeg')

@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    camera_id = request.form['camera_id']
    if camera_id in cameras:
        cameras.remove(camera_id)
    return redirect(url_for('index'))

@app.route('/set_alarm', methods=['POST'])
def set_alarm():
    global alarm_config
    alarm_config['enabled'] = request.form.get('enabled') == 'true'
    alarm_config['target'] = request.form.get('target')
    alarm_config['email'] = request.form.get('email')
    
    # Save config to file
    with open('alarm_config.json', 'w') as f:
        json.dump(alarm_config, f)
    
    return jsonify({'message': 'Alarm settings updated'})

@app.route('/export_detected_faces')
def export_detected_faces():
    if not os.path.exists(EXCEL_FILE_PATH):
        return "No data to export", 204  # No content

    return send_file(EXCEL_FILE_PATH, 
                     download_name='detected_faces.xlsx',  
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    clear_excel_file()
    app.run(debug=True, host='0.0.0.0', port=8000)