import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_svm_model(embeddings_file="embeddings.pickle", model_output_file="svm_model.pickle"):
    # Load the embeddings and labels
    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)
        embeddings = data["embeddings"]
        labels = data["labels"]

    # Encode the labels (convert text labels to numerical labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets (80% training, 20% testing) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # Check if training data contains more than one class
    unique_classes = set(y_train)
    print(f"Unique classes in y_train: {unique_classes}")

    if len(unique_classes) < 2:
        print("Error: Not enough classes in the training set. Please ensure your dataset has samples from at least two different classes.")
        return

    # Create and train the SVM model
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model, label encoder, embeddings, and labels to a file
    with open(model_output_file, "wb") as model_file:
        pickle.dump({
            "svm_model": svm_model, 
            "label_encoder": label_encoder, 
            "embeddings": embeddings, 
            "labels": labels
        }, model_file)
    
    print(f"SVM model, embeddings, and labels saved to {model_output_file}")

if __name__ == "__main__":
    # Train the SVM model using the embeddings and labels
    train_svm_model()
