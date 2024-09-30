import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

data = pd.read_csv('registered_faces.csv')
known_face_names = data['name'].tolist()
known_face_encodings = [np.array(eval(encoding)) for encoding in data['encoding']]

last_detection_times = {name: datetime.min for name in known_face_names}

def detect_faces():
    video_capture = cv2.VideoCapture(1)

    process_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize frame for faster processing
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    current_time = datetime.now()

                    if current_time - last_detection_times[name] > timedelta(seconds=10):
                        print(f"Detected {name} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        last_detection_times[name] = current_time

                face_names.append(name)

        process_frame = not process_frame  

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

detect_faces()
