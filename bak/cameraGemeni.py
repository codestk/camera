import cv2
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def find_camera():
    print("กำลังค้นหากล้องที่เชื่อมต่อ...")
    found_cameras = []
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        for i in range(5):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"พบกล้อง: Index={i}, Backend={backend}, ความละเอียด: {int(width)}x{int(height)}")
                if (i, backend) not in found_cameras:
                    found_cameras.append((i, backend))
                cap.release()
    if not found_cameras:
        return None, None
    return found_cameras[-1]

def nothing(x):
    pass

def run_detection(video_path=None):
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        use_camera = False
    else:
        camera_index, camera_backend = find_camera()
        if camera_index is None:
            messagebox.showerror("Error", "ไม่พบกล้องในระบบ และไม่มีวิดีโอให้เปิด")
            return
        cap = cv2.VideoCapture(camera_index, camera_backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        use_camera = True

    if not cap.isOpened():
        messagebox.showerror("Error", "ไม่สามารถเปิดกล้องหรือวิดีโอได้")
        return

    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 600, 250)
    cv2.createTrackbar("Global_Thresh", "Controls", 125, 255, nothing)
    cv2.createTrackbar("Adaptive_C", "Controls", 2, 20, nothing)
    cv2.createTrackbar("Min_Area", "Controls", 18, 500, nothing)
    cv2.createTrackbar("Max_Area", "Controls", 1500, 5000, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องหรือวิดีโอได้")
            break

        global_thresh_val = cv2.getTrackbarPos("Global_Thresh", "Controls")
        adaptive_c_val = cv2.getTrackbarPos("Adaptive_C", "Controls")
        min_area_val = cv2.getTrackbarPos("Min_Area", "Controls")
        max_area_val = cv2.getTrackbarPos("Max_Area", "Controls")

        result_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 21, adaptive_c_val)
        _, global_thresh = cv2.threshold(gray, global_thresh_val, 255, cv2.THRESH_BINARY_INV)
        final_thresh = cv2.bitwise_and(adaptive_thresh, global_thresh)

        kernel = np.ones((3, 3), np.uint8)
        final_thresh = cv2.erode(final_thresh, kernel, iterations=1)
        final_thresh = cv2.dilate(final_thresh, kernel, iterations=2)
        final_thresh = cv2.erode(final_thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pest_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_val < area < max_area_val:
                pest_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(result_frame, f'Detected: {pest_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detection Result - Camera/Video', result_frame)
        cv2.imshow('Processed Mask', final_thresh)

        if cv2.waitKey(30 if not use_camera else 1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def open_video_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if file_path:
        root.destroy()
        run_detection(file_path)

def start_camera():
    root.destroy()
    run_detection()

# GUI เริ่มต้น
root = tk.Tk()
root.title("เลือกโหมดตรวจจับแมลง")
root.geometry("300x200")

label = tk.Label(root, text="เลือกวิธีการตรวจจับ", font=("Arial", 14))
label.pack(pady=20)

btn1 = tk.Button(root, text="📷 ใช้กล้อง Webcam", command=start_camera, font=("Arial", 12))
btn1.pack(pady=10)

btn2 = tk.Button(root, text="📂 เปิดไฟล์วิดีโอ", command=open_video_file, font=("Arial", 12))
btn2.pack(pady=10)

root.mainloop()
