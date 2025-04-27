# Install dependencies
!pip install ultralytics opencv-python numpy ffmpeg-python

import cv2
import numpy as np
import ffmpeg
import zipfile
import os
from ultralytics import YOLO
from google.colab import files

# Define input video (must be manually uploaded to Colab first)
input_video = "input_video.mp4"  # Change name if needed
frames_dir = "frames"            # Folder to save frames
zip_filename = "labeled_frames.zip"

# Create directory to store frames
os.makedirs(frames_dir, exist_ok=True)

# Load YOLOv8 Pose Model
model = YOLO("yolov8n-pose.pt")

# Open the input video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ ERROR: Cannot open video.")
    exit()

# Ground level for detecting jumps (initialize dynamically)
ground_level = None
jump_threshold = 20  # Pixels above ground level to detect jump
jump_status = "Walking/Standing"

# Function to classify Jumping, Standing, or Sitting
def detect_action(keypoints):
    global ground_level, jump_status

    # Keypoints for lower body
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_knee, right_knee = keypoints[13], keypoints[14]
    left_ankle, right_ankle = keypoints[15], keypoints[16]

    # Convert to NumPy arrays
    left_hip, right_hip = np.array(left_hip), np.array(right_hip)
    left_knee, right_knee = np.array(left_knee), np.array(right_knee)
    left_ankle, right_ankle = np.array(left_ankle), np.array(right_ankle)

    # Compute average positions
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_knee_y = (left_knee[1] + right_knee[1]) / 2
    avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    # Initialize ground level on first frame
    if ground_level is None:
        ground_level = avg_ankle_y

    # Check if the subject is Jumping
    if avg_ankle_y < ground_level - jump_threshold:
        jump_status = "Jumping"
    elif abs(avg_hip_y - avg_knee_y) < 70:
        jump_status = "Sitting"
    else:
        jump_status = "Walking/Standing"

    return jump_status

frame_count = 0

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1

    action = "Walking/Standing"  # Default action

    # Run YOLO Pose Estimation
    results = model(frame)

    detected = False  # Flag to check if any person detected

    for r in results:
        if r.keypoints.xy is not None:
            detected = True
            for kp in r.keypoints.xy:
                keypoints = kp.cpu().numpy()
                action = detect_action(keypoints)

                # Get bounding box
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])

                # Define color based on action
                if action == "Jumping":
                    color = (0, 0, 255)  # Red
                elif action == "Sitting":
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 255, 0)  # Green

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw keypoints
                for (x, y) in keypoints:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    # Save the frame (even if no person detected, it will save as Walking/Standing)
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}_{action}.jpg")
    cv2.imwrite(frame_filename, frame)

# Cleanup
cap.release()
print("✅ Frame processing complete.")

# Create a ZIP file of all the frames
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, _, files_in_dir in os.walk(frames_dir):
        for file in files_in_dir:
            filepath = os.path.join(root, file)
            zipf.write(filepath, arcname=file)

print(f"✅ Zip created: {zip_filename}")

# Download the ZIP
files.download(zip_filename)
