import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image  # Import PIL for image conversion
from model.emotion_cnn import EmotionCNN  # Import your trained model
import time  # For controlling frame rate

# Load the trained model
device = torch.device("cpu")
model = EmotionCNN().to(device)
# Update to load the complete checkpoint instead of just the state dict
checkpoint = torch.load("model/emotion_cnn_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

# Define class labels (7 classes as per your model)
class_mapping = ['Angry','Fear', 'Happy', 'Sad', 'Neutral']

# Update the transform to match the test transform from training
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set FPS control (e.g., 10 frames per second)
fps = 10  # You can adjust this for a slower frame rate if needed
prev_time = time.time()

# Add a history of last N predictions for smoothing
history = []

# Set N for how many frames to use for smoothing
N = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Control the frame rate (to slow down processing)
    current_time = time.time()
    if current_time - prev_time < 1.0 / fps:
        continue
    prev_time = current_time

    # Flip the frame to fix the mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale and detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

    # If faces are detected, process each face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))  # Resize to match model input

            # Convert the NumPy array (face) to a PIL image
            face = Image.fromarray(face)  # Convert NumPy array to PIL Image

            # Apply transformations
            face = transform(face).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

            # Predict emotion
            with torch.no_grad():
                output = model(face)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Convert to probabilities
                confidence, predicted_class = torch.max(probabilities, 0)

            # Store prediction in history
            history.append((predicted_class.item(), confidence.item()))

            # If history length exceeds N, remove oldest prediction
            if len(history) > N:
                history.pop(0)

            # Smooth prediction (use majority voting for class, average confidence)
            class_votes = [x[0] for x in history]
            avg_confidence = sum([x[1] for x in history]) / len(history)

            # Ensure that class_votes is not empty and prevent out-of-range index
            if len(class_votes) > 0:
                # Determine the most common class
                most_common_class = max(set(class_votes), key=class_votes.count)

                # Ensure most_common_class is within range of class_mapping indices
                if 0 <= most_common_class < len(class_mapping):
                    emotion_label = class_mapping[most_common_class]
                else:
                    emotion_label = "Unknown"  # Fallback if index is out of range
            else:
                emotion_label = "Unknown"  # Fallback if no predictions available
                avg_confidence = 0  # Set confidence to zero if no predictions

            confidence_score = avg_confidence * 100

            # Draw rectangle and label on the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion_label} ({confidence_score:.2f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # If no face is detected, display a message
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the webcam feed
    cv2.imshow("Emotion Detection", frame)

    # Exit when ESC key (27) is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
