from flask import Flask, render_template, request, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
import face_recognition
import numpy as np
import os
import cv2
from datetime import datetime

# Flask App and MySQL Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/flask_attendance'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Attendance Model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id', ondelete='CASCADE'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Student Model 
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_file = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)
    attendances = db.relationship('Attendance', backref='student', cascade="all, delete", passive_deletes=True)


# Create the Database and Tables
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']

        if name:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return render_template('add_student.html', error="Could not access the webcam.")
            
            image_count = 0
            embeddings_list = []

            try:
                while image_count < 10:
                    success, frame = camera.read()
                    if not success:
                        return render_template('add_student.html', error="Failed to capture frame from webcam.")
                    
                    # Convert the frame to RGB format
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Find all face locations in the current frame
                    face_locations = face_recognition.face_locations(rgb_frame)

                    if face_locations:
                        # Get the face encodings for each detected face
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        if face_encodings:
                            embeddings_list.append(face_encodings[0])
                            image_count += 1

                            # Optionally, save each captured frame
                            filepath = f'./static/uploads/{name}_{image_count}.jpg'
                            cv2.imwrite(filepath, frame)

                # Compute the average of all embeddings
                avg_embedding = np.mean(np.array(embeddings_list), axis=0)

                # Save the student information to the database
                profile_pic_path = f'./static/uploads/{name}.jpg'
                cv2.imwrite(profile_pic_path, frame)  # Save the last frame as the profile picture

                # Assuming you have a Student model with image_file and embedding fields
                new_student = Student(name=name, image_file=f'{name}.jpg', embedding=avg_embedding.tolist())
                db.session.add(new_student)
                db.session.commit()

                return render_template('add_student.html', success="Student registered successfully!")  # Success alert
            
            finally:
                camera.release()  # Ensure camera release

    return render_template('add_student.html')



def gen_frames():
    """Capture frames from the webcam, detect faces, and display rectangles."""
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)# Open webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame for faster processing (optional)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # Convert from BGR to RGB

            # Find all face locations in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)

            # Loop over face locations and draw rectangles around them
            for (top, right, bottom, left) in face_locations:
                # Scale back up the face locations since we resized the frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw the rectangle around each face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Encode the frame with rectangles and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route to capture video"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    feedback = []  # List to store feedback for multiple faces
    if request.method == 'POST':
        # Open the webcam and capture an image
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            feedback.append("Error: Unable to open the webcam.")
            return render_template('take_attendance.html', feedback=feedback)

        success, frame = camera.read()
        camera.release()  # Ensure the camera is released after capture
        
        if not success:
            feedback.append("Error: Unable to capture image.")
            return render_template('take_attendance.html', feedback=feedback)

        filepath = './static/uploads/temp.jpg'
        cv2.imwrite(filepath, frame)

        # Load the captured image and detect face encodings
        image = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            feedback.append("No face detected.")
            return render_template('take_attendance.html', feedback=feedback)

        # Load student encodings from the database
        students = Student.query.all()
        if not students:
            feedback.append("No student data found in the system.")
            return render_template('take_attendance.html', feedback=feedback)

        student_encodings = [np.array(s.embedding) for s in students]
        threshold = 0.4

        # Process each detected face encoding
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(student_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]

            if best_match_distance < threshold:
                student_id = students[best_match_index].id
                new_attendance = Attendance(student_id=student_id, timestamp=datetime.now())
                db.session.add(new_attendance)
                db.session.commit()
                feedback.append(f"Attendance marked for {students[best_match_index].name}")
            else:
                feedback.append("Unknown face detected.")

        return render_template('take_attendance.html', feedback=feedback)

    return render_template('take_attendance.html', feedback=feedback)


@app.route('/view_attendance')
def view_attendance():
    # Query to fetch attendance records and corresponding student information
    attendance_records = db.session.query(Attendance, Student).join(Student).all()

    # Format timestamp before passing it to the template
    formatted_records = []
    for attendance, student in attendance_records:
        # Format the timestamp as YYYY-MM-DD HH:MM:SS
        formatted_timestamp = attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        formatted_records.append((attendance, student, formatted_timestamp))

    return render_template('view_attendance.html', attendance_records=formatted_records)

@app.route('/view_students')
def view_students():
    # Query all students from the database
    students = Student.query.all()
    return render_template('view_students.html', students=students)

@app.route('/delete_student/<int:id>', methods=['POST'])
def delete_student(id):
    student_to_delete = Student.query.get(id)
    
    if student_to_delete:
        try:
            # Log student details for debugging
            print(f"Deleting student: {student_to_delete.name}")
            
            # Delete the student from the database
            db.session.delete(student_to_delete)
            db.session.commit()
            return redirect(url_for('view_students'))
        except Exception as e:
            db.session.rollback()
            print(f"Error deleting student: {e}")
            return "There was a problem deleting that student."
    else:
        return f"No student found with id {id}"


if __name__ == '__main__':
    app.run(debug=True)
