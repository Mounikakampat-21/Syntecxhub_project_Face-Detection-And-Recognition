import os

print("\n=== STEP 1: Register Person ===")
os.system("python register_person.py")

print("\n=== STEP 2: Training Model ===")
os.system("python train_recognizer.py")

print("\n=== STEP 3: Starting Real-Time Recognition ===")
os.system("python recognize_realtime.py")
