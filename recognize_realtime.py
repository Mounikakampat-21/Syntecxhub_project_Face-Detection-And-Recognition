import cv2
import pickle

haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
TRAINER_PATH = "trainer.yml"
LABELS_PATH = "labels.pickle"

# Load Haar cascade and recognizer
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# Load label IDs
with open(LABELS_PATH, "rb") as f:
    label_ids = pickle.load(f)
# Reverse dict: id -> name
labels = {v: k for k, v in label_ids.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        # Lower conf = better match. Threshold can be tuned.
        if conf < 80:
            name = labels.get(id_, "Unknown")
        else:
            name = "Unknown"

        # Draw bounding box and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f"{name} ({int(conf)})", (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Face Detection & Recognition (OpenCV LBPH)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
