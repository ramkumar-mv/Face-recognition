import cv2
import face_recognition
import numpy as np
import pandas as pd

try:
    data = pd.read_csv('registered_faces.csv')
    known_face_names = data['name'].tolist()
    known_face_encodings = [np.array(eval(encoding)) for encoding in data['encoding']]
except FileNotFoundError:
    known_face_names = []
    known_face_encodings = []

def register_face(name):
    video_capture = cv2.VideoCapture(1)
    face_encodings_list = []

    while len(face_encodings_list) < 5:  
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            face_encodings_list.append(face_encoding)

            if len(face_encodings_list) == 5:
                average_encoding = np.mean(face_encodings_list, axis=0)
                known_face_encodings.append(average_encoding)
                known_face_names.append(name)

                data = pd.DataFrame({'name': known_face_names, 'encoding': [list(enc) for enc in known_face_encodings]})
                data.to_csv('registered_faces.csv', index=False)
                
                print(f"Face registered for {name}")
                video_capture.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

name = input("Enter the name for registration: ")
register_face(name)
