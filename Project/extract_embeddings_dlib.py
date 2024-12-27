import os
import dlib
import cv2
import numpy as np
import pickle

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def extract_embeddings_dlib(original_dataset_path, output_embeddings_file="embeddings.pickle"):
    embeddings_list = []
    labels_list = []

    # Loop through each person in the dataset
    for person_name in os.listdir(original_dataset_path):
        person_folder = os.path.join(original_dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
        
        # Loop through each image of the person
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            faces = face_detector(rgb_image)

            for face in faces:
                # Get the landmarks for the face
                landmarks = shape_predictor(rgb_image, face)

                # Get the face embedding (128-dimensional vector)
                face_embedding = face_recognition_model.compute_face_descriptor(rgb_image, landmarks)
                
                # Append the embedding and label
                embeddings_list.append(np.array(face_embedding))
                labels_list.append(person_name)
                print(f"Extracted embedding from {image_path}")

    # Convert lists to numpy arrays
    embeddings_array = np.array(embeddings_list)
    labels_array = np.array(labels_list)

    # Save embeddings and labels to a file
    with open(output_embeddings_file, "wb") as f:
        pickle.dump({"embeddings": embeddings_array, "labels": labels_array}, f)
    
    print(f"Embeddings saved to {output_embeddings_file}")
    print(embeddings_array)
    print(labels_array)

if __name__ == "__main__":
    # Specify the path to the original dataset
    original_dataset_path = r"C:\Users\sridh\Desktop\mypro\dataset"
    
    # Call the function to extract embeddings
    extract_embeddings_dlib(original_dataset_path)
