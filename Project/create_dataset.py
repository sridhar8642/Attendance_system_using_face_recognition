import cv2
import os

def create_dataset(student_name, num_samples=30, save_path="dataset"):
    # Create directory for the student if it doesn't exist
    student_folder = os.path.join(save_path, student_name)
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    print(f"Capturing {num_samples} samples for {student_name}. Please look at the camera...")

    count = 0  # Counter for captured samples
    while count < num_samples:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w]
            
            # Save the captured image
            img_name = os.path.join(student_folder, f"{student_name}_{count + 1}.jpg")
            cv2.imwrite(img_name, face_roi)
            count += 1
            
            # Display the captured face ROI
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            print(f"Captured sample {count}/{num_samples}")
        
        # Display the frame
        cv2.imshow("Capturing Face Samples", frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Finished capturing {count} samples for {student_name}.")

if __name__ == "__main__":
    student_name = input("Enter the student's name: ").strip()
    create_dataset(student_name)
