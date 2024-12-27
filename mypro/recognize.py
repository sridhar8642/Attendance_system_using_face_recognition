import pickle
import dlib
import cv2
import numpy as np
import mysql.connector

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Define a threshold for recognizing unknown faces
THRESHOLD = 0.4

# MySQL database connection parameters
db_config = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'database': 'face_recognition_db'
}

def store_attendance(name):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO attendance (name) VALUES (%s)", (name,))
        connection.commit()
        print(f"Attendance recorded for: {name}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def recognize_face_dlib(image_path, model_file="svm_model.pickle"):
    # Load the trained SVM model and label encoder
    with open(model_file, "rb") as f:
        data = pickle.load(f)
        svm_model = data["svm_model"]
        label_encoder = data["label_encoder"]
        known_embeddings = data["embeddings"]  # The stored embeddings from training
    
    # Load the image and detect face embeddings
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_image)

    if len(faces) > 0:
        for face in faces:
            # Get landmarks and extract embedding
            landmarks = shape_predictor(rgb_image, face)
            face_embedding = face_recognition_model.compute_face_descriptor(rgb_image, landmarks)
            face_embedding = np.array(face_embedding).reshape(1, -1)

            # Predict the label using the SVM model
            predicted_label_index = svm_model.predict(face_embedding)[0]
            predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

            # Calculate distance between the embedding and known embeddings
            distances = np.linalg.norm(known_embeddings - face_embedding, axis=1)
            min_distance = np.min(distances)

            # If the minimum distance is above the threshold, classify as unknown
            if min_distance > THRESHOLD:
                print(f"Predicted Person: Unknown (min distance: {min_distance:.2f})")
            else:
                print(f"Predicted Person: {predicted_label} (min distance: {min_distance:.2f})")
                store_attendance(predicted_label)  # Record attendance for recognized person
    else:
        print("No face detected in the image")

if __name__ == "__main__":
    # Test the recognition on a new image
    recognize_face_dlib("test_image1.jpg")
