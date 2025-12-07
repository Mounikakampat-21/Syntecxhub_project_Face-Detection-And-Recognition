import cv2
import os

haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def register_person(person_name, num_images=30):
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    cap = cv2.VideoCapture(0)
    count = 0

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
            # Draw rectangle on screen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop face and save
            face_img = gray[y:y + h, x:x + w]
            img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1
            print(f"Saved {img_path}")

            if count >= num_images:
                break

        cv2.putText(frame, f"Person: {person_name} Images: {count}/{num_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Register Person - Press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done capturing faces.")

if __name__ == "__main__":
    name = input("Enter person's name: ").strip()
    register_person(name, num_images=30)
