from flask import Flask, request, render_template, redirect, url_for, flash, Response,send_file
import os
import cv2
import numpy as np
import time
import pickle
from imutils import paths
import dlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from datetime import datetime
import csv
import mysql.connector

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'attendance_system'
}

# Function to get a database connection
def get_db_connection():
    return mysql.connector.connect(**db_config)


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Directory and file paths
DATASET_DIR = 'dataset'
MODEL_PATH = "model"
OUTPUT_PATH = "output"
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMBEDDING_FILE = os.path.join(OUTPUT_PATH, "embeddings.pickle")
RECOGNIZER_FILE = os.path.join(OUTPUT_PATH, "recognizer.pickle")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_PATH, "le.pickle")
ATTENDANCE_FILE = "attendance.csv"
SHAPE_PREDICTOR = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
FACE_RECOGNITION_MODEL = os.path.join(MODEL_PATH, "dlib_face_recognition_resnet_model_v1.dat")
CONFIDENCE_THRESHOLD = 0.5

# Ensure necessary directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
embedder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)

#route for creating dataset
@app.route('/create_dataset', methods=['GET', 'POST'])
def create_dataset():
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']

        #add students as registered in db
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Save student info in the registered_students table
        cursor.execute("INSERT INTO registered_students (name, roll_number) VALUES (%s, %s)", (name, roll_number))
        conn.commit()
        
        cursor.close()
        conn.close()

        # Define path to save images for this person
        person_path = os.path.join(DATASET_DIR, name)
        os.makedirs(person_path, exist_ok=True)

        # Start capturing images
        cam = cv2.VideoCapture(0)
        total = 0

        while total < 50:
            ret, frame = cam.read()
            if not ret:
                break

            # Convert frame to RGB for better compatibility with dlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector(rgb_frame)

            for face in faces:
                # Get the bounding box coordinates
                x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                # Draw the green rectangle
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # Display the total count inside the bounding box
                text = f"Captured: {total}/50"
                cv2.putText(
                    frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2, cv2.LINE_AA
                )

                # Crop the face region with padding
                face_img = frame[y:y2, x:x2]

                # Resize the cropped image to a standard size (e.g., 400x400 pixels)
                passport_face = cv2.resize(face_img, (400, 400), interpolation=cv2.INTER_AREA)

                # Save the processed face image
                img_path = os.path.join(person_path, f"{str(total).zfill(5)}.png")
                cv2.imwrite(img_path, passport_face)
                total += 1
                print(f"Captured image {total} / 50")  # Print count to terminal
                break  # Capture only one face per frame

            # Show the frame with the green rectangle
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Delay of 1 second after capturing each image
            if total < 50:  # To avoid extra delay after the last image
                time.sleep(1)

        cam.release()
        cv2.destroyAllWindows()

        flash('Dataset creation successful!', 'success')
        return redirect(url_for('index'))

    return render_template('create_dataset.html')


# Route for preprocessing embeddings
@app.route('/preprocess_embeddings', methods=['POST'])
def preprocess_embeddings():
    image_paths = list(paths.list_images(DATASET_DIR))
    
    known_embeddings = []
    known_names = []
    total = 0

    for (i, image_path) in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}")
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = detector(rgb_image, 1)

        for box in boxes:
            shape = predictor(rgb_image, box)
            face_descriptor = embedder.compute_face_descriptor(rgb_image, shape)
            known_names.append(name)
            known_embeddings.append(np.array(face_descriptor))
            total += 1

    data = {"embeddings": known_embeddings, "names": known_names}
    with open(EMBEDDING_FILE, "wb") as f:
        f.write(pickle.dumps(data))

    flash(f'Preprocessed {total} embeddings successfully!', 'success')
    return redirect(url_for('index'))

# Route for training the SVM model
@app.route('/train_model', methods=['POST'])
def train_model():
    if not os.path.exists(EMBEDDING_FILE):
        flash('Embeddings file not found. Please preprocess embeddings first.', 'danger')
        return redirect(url_for('index'))

    # Load the embeddings data
    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    known_names = data["names"]
    
    # Check if we have multiple classes
    if len(set(known_names)) < 2:
        flash('Error: The dataset contains less than 2 classes. Please add more data.', 'danger')
        return redirect(url_for('index'))

    # Proceed with label encoding and training
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(known_names)

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # Save the trained recognizer and label encoder
    with open(RECOGNIZER_FILE, "wb") as f:
        f.write(pickle.dumps(recognizer))
    with open(LABEL_ENCODER_FILE, "wb") as f:
        f.write(pickle.dumps(label_encoder))

    flash('Model training completed successfully!', 'success')
    return redirect(url_for('index'))

# Store recognized names for attendance
recognized_names = set()

# Function to generate video frames for streaming
def generate_frames():
    cam = cv2.VideoCapture(0)

    # Load the recognizer and label encoder if available
    recognizer = None
    le = None
    if os.path.exists(RECOGNIZER_FILE) and os.path.exists(LABEL_ENCODER_FILE):
        with open(RECOGNIZER_FILE, "rb") as f:
            recognizer = pickle.load(f)
        with open(LABEL_ENCODER_FILE, "rb") as f:
            le = pickle.load(f)

    if recognizer is None or le is None:
        print("Error: Recognizer or label encoder not found. Please train the model first.")
        exit(1)

    global recognized_names  # Track recognized names globally

    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector(rgb_frame)

            for box in boxes:
                shape = predictor(rgb_frame, box)
                face_embedding = embedder.compute_face_descriptor(rgb_frame, shape)
                preds = recognizer.predict_proba([np.array(face_embedding)])[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j] if proba >= CONFIDENCE_THRESHOLD else "Unknown"

                if name != "Unknown":
                    # Check attendance status
                    if name not in recognized_names:
                        status = log_attendance(name)
                        recognized_names.add(name)
                    else:
                        status = "already_marked"

                    # Prepare display text
                    if status == "marked":
                        text = f"{name}:{proba * 100:.2f}% - Attendance Marked"
                    else:
                        text = f"{name}:{proba * 100:.2f}% - Attendance Already Marked"
                else:
                    text = "Unknown"

                # Draw bounding box and label
                (startX, startY, endX, endY) = (box.left(), box.top(), box.right(), box.bottom())
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route to display webcam stream
@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to log attendance into MySQL database
def log_attendance(name):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get today's date
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check if attendance is already marked for today
    sql_check = "SELECT * FROM attendance WHERE name = %s AND DATE(timestamp) = %s"
    cursor.execute(sql_check, (name, today_date))
    result = cursor.fetchone()

    if result:
        status = "already_marked"
    else:
        # Insert attendance record
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql_insert = "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)"
        cursor.execute(sql_insert, (name, timestamp))
        conn.commit()
        status = "marked"

    cursor.close()
    conn.close()
    return status


@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('view_attendance.html', records=records)

@app.route('/view_students')
def view_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch all registered students
    cursor.execute("SELECT * FROM registered_students")  # Adjust table name and columns based on your database
    students = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('view_students.html', students=students)



@app.route('/export_attendance')
def export_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query attendance records
    cursor.execute("SELECT name, DATE_FORMAT(timestamp, '%Y-%m-%d %H:%i:%s') AS formatted_date FROM attendance")
    rows = cursor.fetchall()

    # Define the CSV file name
    csv_file_path = 'attendance.csv'

    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Timestamp'])
        writer.writerows(rows)

    cursor.close()
    conn.close()

    # Send the file for download
    return send_file(csv_file_path, as_attachment=True)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
