# 🕺 Human Activity Recognition using YOLOv8 Pose

This project uses **YOLOv8 Pose Estimation** to detect and classify basic human actions — **Jumping**, **Sitting**, and **Standing** — from a video input.

🔍 It works by detecting key points using `yolov8n-pose.pt` and applying simple geometry-based heuristics to identify the current action. The results are overlaid on the video and saved as an output file.

---

## 📁 Files

- **Main.ipynb** – Core notebook that performs action detection and outputs the processed video
- **README.md** – You're here!

---

## 🚀 How to Use (in Google Colab)

1. Download the file `Main.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload the notebook.
4. Upload your input video (e.g., `all.mp4`) in the Colab environment.
5. Run all the cells — the output will be a downloadable, processed video with action annotations.

---

## 📦 Dependencies

The notebook installs the following packages automatically:

- `ultralytics` (for YOLOv8)
- `opencv-python` (for video processing)
- `numpy`
- `ffmpeg-python` (for video compression)

---

## 🧠 How It Works

1. **Pose Estimation**: Uses `yolov8n-pose.pt` to detect keypoints on the human body.
2. **Action Detection**: 
   - **Jumping** is detected if the ankle key points are significantly above the ground level.
   - **Sitting** is detected if the hip and knee key points are close together.
   - Otherwise, the person is assumed to be **Standing**.
3. **Visualization**: Bounding boxes and action labels are drawn on each frame.
4. **Output**: The annotated video is compressed and made available for download.

---

## 📸 Sample Output

<p float="left">
  <img src="https://github.com/user-attachments/assets/e43b7ade-7f4d-41c7-90db-7fe6f58567a2" width="30%" />
  <img src="https://github.com/user-attachments/assets/4010e2d0-31f8-4bda-b562-911937d48ccd" width="30%" />
  <img src="https://github.com/user-attachments/assets/cbba77c2-3d3c-48d0-8775-754b3219f881" width="30%" />
</p>


## 🏫 Author

**Reet Singh Solanki**  
Indian Institute of Technology Bombay  
2021 Batch — Electrical Engineering  
