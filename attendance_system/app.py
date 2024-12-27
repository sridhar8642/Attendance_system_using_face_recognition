from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import face_recognition
import numpy as np
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Load embeddings and names
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")
else:
    embeddings = []

if os.path.exists("names.json"):
    with open("names.json", 'r') as f:
        names = json.load(f)
else:
    names = {}

attendance_file = 'attendance.json'
if os.path.exists(attendance_file):
    with open(attendance_file, 'r') as f:
        attendance = json.load(f)
else:
    attendance = {}

# Route to home
@app.route('/')
def home():
    return render_template('home.html')

# Route to add a new student
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['file']
        if name and file:
            filename = f"{name}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image and compute the embeddings
            image = face_recognition.load_image_file(filepath)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                embeddings.append(encoding[0])
                names[filename] = name
                np.save('embeddings.npy', np.array(embeddings))
                with open("names.json", 'w') as f:
                    json.dump(names, f)
                return redirect(url_for('home'))
    return render_template('add_student.html')

# Route to take attendance
@app.route('/take_attendance')
def take_attendance():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(embeddings, face_encoding)

            if True in matches:
                index = matches.index(True)
                image_file = list(names.keys())[index]
                name = names[image_file]

                if name not in attendance:
                    attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(attendance_file, 'w') as f:
                        json.dump(attendance, f)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Taking Attendance", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('view_attendance'))

# Route to view attendance
@app.route('/view_attendance')
def view_attendance():
    with open(attendance_file, 'r') as f:
        attendance_log = json.load(f)

    df = pd.DataFrame(list(attendance_log.items()), columns=['Name', 'Timestamp'])
    df.to_html('static/attendance_log.html', index=False)
    return render_template('view_attendance.html')

if __name__ == '__main__':
    app.run(debug=True)
