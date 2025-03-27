import face_recognition
import os
import cv2
import numpy as np
import webbrowser
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to load known face encodings and names
def load_known_faces(images_folder):
    known_face_encodings = []
    known_face_names = []
    image_files = os.listdir(images_folder)

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        img = face_recognition.load_image_file(image_path)

        # Detect and encode the face in the image
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:  # Check if at least one face was found
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(image_file)[0])  # Use the filename as the label

    return known_face_encodings, known_face_names


# Load known faces from the "images" folder
images_folder = "images"
known_face_encodings, known_face_names = load_known_faces(images_folder)
print(f"Loaded {len(known_face_encodings)} known faces.")

# Prompt user to upload a photo for comparison
print("Please select a photo to upload for face recognition.")
Tk().withdraw()  # Hide the root window
uploaded_photo_path = askopenfilename(title="Select a Photo", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

if not uploaded_photo_path:
    print("No photo selected. Exiting.")
else:
    print(f"Processing uploaded photo: {uploaded_photo_path}")

    # Load the uploaded photo
    uploaded_photo = face_recognition.load_image_file(uploaded_photo_path)

    # Detect and encode faces in the uploaded photo
    uploaded_face_locations = face_recognition.face_locations(uploaded_photo)
    uploaded_face_encodings = face_recognition.face_encodings(uploaded_photo, uploaded_face_locations)

    if not uploaded_face_encodings:
        print("No faces detected in the uploaded photo.")
    else:
        # Compare uploaded photo faces with known faces
        for (top, right, bottom, left), uploaded_face_encoding in zip(uploaded_face_locations, uploaded_face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, uploaded_face_encoding, tolerance=0.5)
            name = "Unknown"
            confidence = 100.0  # Default confidence for unknown faces

            if True in matches:
                # Get the index of the first match
                match_index = matches.index(True)
                name = known_face_names[match_index]

                # Calculate confidence using distance (lower distance = higher confidence)
                face_distance = face_recognition.face_distance([known_face_encodings[match_index]], uploaded_face_encoding)[0]
                confidence = max(0, 100 - face_distance * 100)

                # Check if a PDF with the same name exists
                pdf_filename = f"{name}.pdf"
                if os.path.exists(pdf_filename):
                    print(f"Opening PDF: {pdf_filename}")
                    webbrowser.open(pdf_filename)
                else:
                    print(f"PDF for {name} not found.")
            else:
                print("Face not recognized.")

            # Annotate the face for visualization
            cv2.rectangle(uploaded_photo, (left, top), (right, bottom), (0, 255, 0), 2)
            text = f"{name} ({confidence:.1f}%)"
            cv2.putText(uploaded_photo, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save and display the annotated image
        annotated_image_path = "annotated_image.jpg"
        cv2.imwrite(annotated_image_path, cv2.cvtColor(uploaded_photo, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved as {annotated_image_path}")
        cv2.imshow("Uploaded Photo", cv2.cvtColor(uploaded_photo, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
