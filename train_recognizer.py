import os
import cv2
import numpy as np
import pickle

DATASET_DIR = "dataset"
TRAINER_PATH = "trainer.yml"
LABELS_PATH = "labels.pickle"

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                path = os.path.join(root, file)
                label = os.path.basename(root)  # folder name = person name

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]

                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                x_train.append(image)
                y_labels.append(id_)

    if len(x_train) == 0:
        print("No images found in dataset. Please register persons first.")
        return

    # Train the recognizer
    recognizer.train(x_train, np.array(y_labels))
    recognizer.write(TRAINER_PATH)

    # Save label mapping (id -> name)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_ids, f)

    print("[INFO] Training complete.")
    print(f"[INFO] Model saved to {TRAINER_PATH}")
    print(f"[INFO] Labels saved to {LABELS_PATH}")

if __name__ == "__main__":
    train_recognizer()
