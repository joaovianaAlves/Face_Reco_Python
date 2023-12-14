import face_recognition
import os
import cv2
import numpy as np
import sys
import math

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings, self.known_face_names = self.load_known_faces()
        self.frame_count = 0
        self.process_current_frame = True
        self.recognized_face_window_name = 'Recognized Face'
        self.video_capture = cv2.VideoCapture(0)
        self.init_window()

    def init_window(self):
        cv2.namedWindow(self.recognized_face_window_name, cv2.WINDOW_NORMAL)

    def load_known_faces(self):
        known_face_encodings = []
        known_face_names = []

        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            known_face_encodings.append(face_encoding)
            known_face_names.append(image)

        print(known_face_names)
        return known_face_encodings, known_face_names

    def display_known_faces(self):
        for i, face_name in enumerate(self.known_face_names):
            known_image = cv2.imread(f'faces/{face_name}')
            cv2.imshow(f'Known Face {i + 1}', known_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def draw_face_rectangles(self, frame, faces):
        for (top, right, bottom, left), name in faces:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)

            # Display the name and confidence inside the rectangle
            text = f'{name.split()[0]}\n{float(name.split("(")[1].split("%")[0]):.2f}%'
            cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255))

    def process_frame(self, frame):
        self.frame_count += 1

        if self.frame_count % 3 == 0:  # Process every 3 frames
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = 'Unknown'
                confidence = 'Unknown'

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = self.face_confidence(face_distances[best_match_index])

                    # Get the recognized face image
                    recognized_face_path = f'faces/{name}'
                    print(f"Loading recognized face image from: {recognized_face_path}")

                    recognized_face_image = cv2.imread(recognized_face_path)
                    if recognized_face_image is not None:
                        # Resize the recognized face image to match the frame height
                        h, w, _ = frame.shape
                        recognized_face_image = cv2.resize(recognized_face_image, (w, h))

                        # Add the percentage confidence to the recognized face image
                        confidence_text = f'Confidence: {confidence}'
                        cv2.putText(
                            recognized_face_image, confidence_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )

                        # Combine the frame and recognized face side by side
                        combined_image = np.concatenate((frame, recognized_face_image), axis=1)

                        face_names.append(f'{name} ({confidence})')

                        # Draw rectangles and display name and confidence inside
                        self.draw_face_rectangles(combined_image, zip(face_locations, face_names))

                        # Display the combined image
                        cv2.imshow(self.recognized_face_window_name, combined_image)

    def face_confidence(self, face_distance, face_match_threshold=0.6):
        range_ = (1.0 - face_match_threshold)
        linear_value = (1.0 - face_distance) / (range_ * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_value * 100, 2)) + '%'
        else:
            value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

    def run_recognition(self):
        if not self.video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = self.video_capture.read()

            if not ret:
                print("Error reading frame. Exiting...")
                break

            if self.process_current_frame:
                self.process_frame(frame)

            self.process_current_frame = not self.process_current_frame

            if cv2.waitKey(1) == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
    fr.display_known_faces()
