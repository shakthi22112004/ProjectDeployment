import cv2
import face_recognition
import os
import numpy as np
import webbrowser

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

# To keep track of opened PDFs
opened_pdfs = set()

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

            # Check if a PDF with the same name exists and has not been opened yet
            pdf_filename = f"{name}.pdf"
            if os.path.exists(pdf_filename) and pdf_filename not in opened_pdfs:
                print(f"Opening PDF: {pdf_filename}")
                webbrowser.open(pdf_filename)
                opened_pdfs.add(pdf_filename)  # Mark this PDF as opened

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name and confidence
        text = f"{name} ({confidence:.1f}%)"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if name == "Unknown":
            # Draw a rectangle and label for unknown faces
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the frame with annotations
    cv2.imshow("Face Recognition", frame)

    # Capture the image when 'c' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Press 'c' to capture the current frame
        capture_filename = "captured_image.jpg"
        cv2.imwrite(capture_filename, frame)
        print(f"Image captured and saved as {capture_filename}")
    elif key == 27:  # Press ESC to exit
        print("Exiting program.")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
