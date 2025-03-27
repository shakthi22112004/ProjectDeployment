import cv2
import face_recognition
import os
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Load images from the "images" folder
images_folder = "images"
image_files = os.listdir(images_folder)

# Create lists for known face encodings and their labels
known_face_encodings = []
known_face_names = []

# Load and encode each image in the folder
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    img = face_recognition.load_image_file(image_path)
    
    # Detect and encode the face in the image
    face_encodings = face_recognition.face_encodings(img)
    if face_encodings:  # Check if at least one face was found
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(os.path.splitext(image_file)[0])  # Use the filename as the label

print(f"Loaded {len(known_face_encodings)} known faces.")

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect all faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        confidence = 100.0  # Default confidence for unknown faces

        if True in matches:
            # Get the index of the first match
            match_index = matches.index(True)
            name = known_face_names[match_index]
            
            # Calculate confidence using distance (lower distance = higher confidence)
            face_distance = face_recognition.face_distance([known_face_encodings[match_index]], face_encoding)[0]
            confidence = max(0, 100 - face_distance * 100)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name and confidence
        text = f"{name} ({confidence:.1f}%)"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame with annotations
    cv2.imshow("Face Recognition", frame)

    # Break the loop if the user presses ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
