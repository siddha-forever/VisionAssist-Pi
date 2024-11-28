import cv2
import os
import time
from datetime import datetime
from picamera2 import Picamera2
from espeak import espeak
import face_recognition
import numpy as np
import RPi.GPIO as GPIO
import pickle
from imutils import paths

# Global variables for GPIO
TRIG = 27
ECHO = 22

# Function to initialize GPIO pins for ultrasonic sensor
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

# Function to read distance from the ultrasonic sensor
def read_distance():
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.00001)
    GPIO.output(TRIG, GPIO.HIGH)
    GPIO.output(TRIG, GPIO.LOW)

    pulse_start = time.time()
    while GPIO.input(ECHO) == GPIO.LOW:
        pulse_start = time.time()

    pulse_end = pulse_start
    while GPIO.input(ECHO) == GPIO.HIGH:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = (34300 * pulse_duration) / 2

    if distance <= 400:
        return distance
    else:
        return None

# Function to provide voice feedback
def give_feedback(distance):
    espeak.set_voice("en")
    espeak.set_voice("whisper")
    espeak.synth("Obstacle is nearby")
    espeak.synth(str(round(distance, 1)))

# Function to capture photos and save them for training
def capture_photos(person_name):
    dataset_folder = "dataset"
    person_folder = os.path.join(dataset_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    time.sleep(2)
    print(f"Taking photos for {person_name}. Press SPACE to capture, 'q' to quit.")
    photo_count = 0

    while True:
        frame = picam2.capture_array()
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(person_folder, f"{person_name}_{timestamp}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

# Function to train the face recognition model
def train_model():
    print("[INFO] Start processing faces...")
    imagePaths = list(paths.list_images("dataset"))
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    data = {"encodings": knownEncodings, "names": knownNames}
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)
    print("[INFO] Training complete. Encodings saved.")

# Function to perform real-time face recognition
def recognize_faces():
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
    picam2.start()

    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    picam2.stop()

# Main function to run all components
def main():
    setup_gpio()
    print("Press Ctrl+C to stop.")

    try:
        while True:
            distance = read_distance()
            if distance:
                print(f"Distance: {distance:.1f} cm")
                if distance <= 18:
                    give_feedback(distance)

            recognize_faces()

    except KeyboardInterrupt:
        print("Program stopped by user.")
        GPIO.cleanup()

if __name__ == "__main__":
    main()