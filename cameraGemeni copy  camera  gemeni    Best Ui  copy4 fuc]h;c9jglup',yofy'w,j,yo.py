import sys
import os
import json
import time
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QMessageBox, QFileDialog, QComboBox, QSpinBox, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
import threading
import cv2
import numpy as np
try:
    import winsound # For playing sound on Windows
except ImportError:
    print("Warning: 'winsound' module not found. Sound alerts will be disabled. (This is expected on non-Windows systems)")
    winsound = None


# ======================================================================
# Main Settings
# ======================================================================
SAVE_DIR = "captures"
SETTINGS_FILE = "settings.json"
os.makedirs(SAVE_DIR, exist_ok=True)

# Map readable backend name -> OpenCV constant
BACKENDS = {
    "Auto": None,
    "DirectShow (DSHOW)": cv2.CAP_DSHOW,
    "Media Foundation (MSMF)": cv2.CAP_MSMF,
}

# ======================================================================
# Video Processing Worker Thread
# ======================================================================
class VideoWorker(QObject):
    connection_failed = pyqtSignal(str)

    def __init__(self, source_type="camera", video_path=None, params=None,
                 cam_index=0, cam_backend_name="Auto",
                 target_fps=30, 
                 session_save_path="captures",
                 session_timestamp=""):
        super().__init__()
        self.running = True
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.capture_snapshot = False
        self.params = params or {
            "global_thresh": 125, "adaptive_c": 2, "adaptive_block_idx": 9,
            "min_area": 18, "max_area": 1500, "erode_iters": 1,
            "dilate_iters": 2, "kernel_size": 3
        }
        self.lock = threading.Lock()
        self.latest_result_frame = None
        self.latest_mask_frame = None
        self.latest_status_text = "Initializing..."

        self.source_type = source_type
        self.video_path = video_path
        self.cam_index = cam_index
        self.cam_backend_name = cam_backend_name
        self.target_fps = target_fps

        # New state variables for auto-save and sound alerts
        self.auto_save_on_detect = False
        self.play_sound_on_detect = False
        self.last_event_time = 0
        self.event_cooldown_seconds = 3.0 # Cooldown 3 วินาที

        # Session management variables
        self.session_save_path = session_save_path
        self.session_timestamp = session_timestamp
        self.detection_log = [] # List to store (timestamp, count) tuples

    # -------------------- main thread loop --------------------
    def run(self):
        cap = None
        try:
            if self.source_type == "video":
                if not self.video_path or not os.path.exists(self.video_path):
                    self.connection_failed.emit("ไม่พบไฟล์วิดีโอหรือพาธไม่ถูกต้อง")
                    return
                cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.connection_failed.emit(
                        f"เปิดไฟล์วิดีโอไม่สำเร็จ (FFMPEG):\n{self.video_path}\n\nแนะนำเข้ารหัสเป็น H.264 (.mp4) ถ้ายังเปิดไม่ได้")
                    return
                self.print_video_props(cap)
                print(f"[SOURCE] Using video file (FFMPEG): {self.video_path}")
            else:
                backend = BACKENDS.get(self.cam_backend_name)
                if backend is None:
                    for b in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                        cap = cv2.VideoCapture(self.cam_index, b)
                        if cap.isOpened():
                            print(f"[SOURCE] Using camera index {self.cam_index} (backend {b})")
                            break
                    if cap is not None and not cap.isOpened():
                        cap.release(); cap = None
                    if cap is None:
                        self.connection_failed.emit(
                            f"ไม่พบหรือเปิดกล้องไม่ได้ (index={self.cam_index})\nลองสลับ backend เป็น DSHOW หรือ MSMF")
                        return
                else:
                    cap = cv2.VideoCapture(self.cam_index, backend)
                    if not cap.isOpened():
                        self.connection_failed.emit(
                            f"เปิดกล้องไม่สำเร็จ index={self.cam_index} backend={self.cam_backend_name}")
                        return
                print(f"[SOURCE] Using camera index {self.cam_index} ({self.cam_backend_name})")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                if self.target_fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

            frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
            fps_avg, n, last_t = 0.0, 0, time.time()

            while self.running:
                if self.paused:
                    QThread.msleep(100)
                    continue

                start_t = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or cannot read frame.")
                    break

                result_frame, mask_frame, pest_count, mean_area = self.process_frame(frame)
                
                # Added auto-save and sound alert logic
                if pest_count > 0:
                    current_time = time.time()
                    if current_time - self.last_event_time > self.event_cooldown_seconds:
                        
                        ts_for_log = datetime.now()
                        self.detection_log.append((ts_for_log, pest_count))

                        # 1. Auto-save if enabled
                        if self.auto_save_on_detect:
                            ts_str = ts_for_log.strftime('%Y%m%d_%H%M%S_%f')[:-3]
                            filename = os.path.join(self.session_save_path, f"auto_detect_{ts_str}.png")
                            cv2.imwrite(filename, result_frame)
                            print(f"[AUTO-SAVE] Image saved to {filename}")

                        # 2. Play sound if enabled
                        if self.play_sound_on_detect:
                            if winsound:
                                try:
                                    winsound.Beep(1000, 200)
                                except Exception as e:
                                    print(f"[ERROR] Could not play beep sound via winsound: {e}. Trying fallback.")
                                    print('\a', flush=True)
                            else:
                                print('\a', flush=True)

                        self.last_event_time = current_time

                now = time.time()
                fps = 1.0 / max(1e-6, (now - last_t)); last_t = now; n += 1
                fps_avg = (fps_avg * (n - 1) + fps) / n
                status_text = (
                    f"Detected: {pest_count} | FPS: {fps:.1f} (avg {fps_avg:.1f}) | Mean Area: {mean_area:.0f}")

                if self.capture_snapshot:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(os.path.join(self.session_save_path, f"snap_{ts}_detected.png"), result_frame)
                    cv2.imwrite(os.path.join(self.session_save_path, f"snap_{ts}_original.png"), frame)
                    self.capture_snapshot = False

                if self.recording and self.video_writer is not None:
                    self.video_writer.write(result_frame)

                with self.lock:
                    self.latest_result_frame = result_frame
                    self.latest_mask_frame = mask_frame
                    self.latest_status_text = status_text

                elapsed = time.time() - start_t
                if frame_interval > 0 and frame_interval > elapsed:
                    time.sleep(frame_interval - elapsed)
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            if cap is not None:
                cap.release()
            self.write_summary_file()
            print("Video thread stopped.")

    def write_summary_file(self):
        if not self.detection_log:
            print("[SUMMARY] No detections were logged in this session.")
            return

        summary_path = os.path.join(self.session_save_path, f"summary_{self.session_timestamp}.txt")
        total_events = len(self.detection_log)
        total_objects = sum(count for _, count in self.detection_log)
        avg_objects = total_objects / total_events if total_events > 0 else 0

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*40 + "\n")
                f.write(f"สรุปผลการตรวจจับ (Session: {self.session_timestamp})\n")
                f.write("="*40 + "\n\n")
                f.write(f"จำนวนครั้งที่ตรวจเจอ (Events): {total_events} ครั้ง\n")
                f.write(f"จำนวนวัตถุที่เจอทั้งหมด (Objects): {total_objects} ชิ้น\n")
                f.write(f"จำนวนวัตถุเฉลี่ยต่อครั้ง: {avg_objects:.2f} ชิ้น\n\n")
                f.write("="*40 + "\n")
                f.write("ประวัติการตรวจจับ (เวลา: จำนวนที่เจอ)\n")
                f.write("="*40 + "\n")
                for ts, count in self.detection_log:
                    f.write(f"{ts.strftime('%H:%M:%S.%f')[:-3]}: {count} ชิ้น\n")
            print(f"[SUMMARY] Summary file saved to: {summary_path}")
        except Exception as e:
            print(f"[ERROR] Could not write summary file: {e}")


    def process_frame(self, frame):
        result_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gT = self.params["global_thresh"]
        C = self.params["adaptive_c"]
        blkI = self.params["adaptive_block_idx"]
        ksz = self.params["kernel_size"]
        er = self.params["erode_iters"]
        di = self.params["dilate_iters"]
        mnA = self.params["min_area"]
        mxA = self.params["max_area"]
        block = 2 * blkI + 3
        ksz = max(1, ksz | 1)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block, C)
        _, global_t = cv2.threshold(gray, gT, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(adaptive, global_t)
        kernel = np.ones((ksz, ksz), np.uint8)
        if er > 0: mask = cv2.erode(mask, kernel, iterations=er)
        if di > 0: mask = cv2.dilate(mask, kernel, iterations=di)
        if er > 0: mask = cv2.erode(mask, kernel, iterations=er)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count, areas = 0, []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if mnA < area < mxA:
                count += 1
                areas.append(area)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        mean_area = (sum(areas) / len(areas)) if areas else 0
        return result_frame, mask, count, mean_area

    def toggle_recording(self, frame_size):
        self.recording = not self.recording
        if self.recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_name = os.path.join(self.session_save_path, f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            fps_for_rec = 20.0
            self.video_writer = cv2.VideoWriter(out_name, fourcc, fps_for_rec, frame_size)
            print(f"[REC] Started recording -> {out_name} @ {fps_for_rec} fps")
            return True, out_name
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print("[REC] Stopped recording")
            return False, ""

    @staticmethod
    def print_video_props(cap):
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[VIDEO] {w}x{h} @ {fps:.2f} fps, frames={total}")
        except Exception:
            pass

# ======================================================================
# Main Application Window (UI)
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("โปรแกรมตรวจจับแมลง – Camera/Video")
        self.setGeometry(100, 100, 1700, 920)

        main_widget = QWidget(); main_layout = QHBoxLayout(); main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Set default values first
        self.params = {
            "global_thresh": 125, "adaptive_c": 2, "adaptive_block_idx": 9,
            "min_area": 18, "max_area": 1500, "erode_iters": 1,
            "dilate_iters": 2, "kernel_size": 3
        }
        self.cam_backend_name = "Auto"
        self.cam_index = 0
        self.target_fps = 30
        
        # <<< MODIFIED: Initialize worker and thread to None *before* creating controls
        self.video_thread = None
        self.video_worker = None

        self.load_settings()

        controls_layout = self.create_controls_panel()
        main_layout.addLayout(controls_layout, 1)
        video_layout = self.create_video_panel()
        main_layout.addLayout(video_layout, 4)

        self.update_controls_from_settings()

        self.start_video_thread(source_type="camera", video_path=None)

        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_gui_from_worker)
        self.ui_update_timer.start(33)
        self.set_stylesheet()

    def start_video_thread(self, source_type: str, video_path: str | None):
        self.stop_video_thread()

        session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_save_path = os.path.join(SAVE_DIR, session_timestamp)
        os.makedirs(session_save_path, exist_ok=True)
        print(f"[SESSION] New session started. Saving files to: {session_save_path}")

        self.video_worker = VideoWorker(
            source_type=source_type,
            video_path=video_path,
            params=self.params.copy(),
            cam_index=self.cam_index,
            cam_backend_name=self.cam_backend_name,
            target_fps=self.target_fps,
            session_save_path=session_save_path,
            session_timestamp=session_timestamp,
        )
        self.video_thread = QThread()
        self.video_worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.connection_failed.connect(self.show_connection_error)
        self.video_thread.start()

    def stop_video_thread(self):
        if self.video_thread is not None and self.video_worker is not None:
            self.video_worker.running = False
            self.video_thread.quit(); self.video_thread.wait()
            self.video_worker.deleteLater(); self.video_thread.deleteLater()
        self.video_worker = None; self.video_thread = None

    def create_controls_panel(self):
        layout = QVBoxLayout(); layout.setSpacing(10); layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        src_row = QHBoxLayout()
        self.cmb_backend = QComboBox(); self.cmb_backend.addItems(list(BACKENDS.keys()))
        self.cmb_backend.currentTextChanged.connect(self.on_backend_changed)
        src_row.addWidget(QLabel("Backend")); src_row.addWidget(self.cmb_backend)

        self.spin_cam_idx = QSpinBox(); self.spin_cam_idx.setRange(0, 10)
        self.spin_cam_idx.valueChanged.connect(self.on_cam_index_changed)
        src_row.addWidget(QLabel("Cam Index")); src_row.addWidget(self.spin_cam_idx)
        layout.addLayout(src_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Target FPS"))
        self.spin_fps = QSpinBox(); self.spin_fps.setRange(1, 120)
        self.spin_fps.valueChanged.connect(self.on_fps_changed)
        fps_row.addWidget(self.spin_fps)
        layout.addLayout(fps_row)

        self.btn_open_video = QPushButton("เลือกไฟล์วิดีโอ…"); self.btn_open_video.clicked.connect(self.open_video_file); layout.addWidget(self.btn_open_video)
        self.btn_use_camera = QPushButton("ใช้กล้อง (Webcam)"); self.btn_use_camera.clicked.connect(lambda: self.switch_source("camera", None)); layout.addWidget(self.btn_use_camera)

        self.sliders = {
            "global_thresh": self.create_slider("global_thresh", "เกณฑ์ความดำ", 0, 255, self.params["global_thresh"]),
            "adaptive_c": self.create_slider("adaptive_c", "ความไวแสง", 1, 20, self.params["adaptive_c"]),
            "adaptive_block_idx": self.create_slider("adaptive_block_idx", "ขนาด Block (Adaptive)", 1, 49, self.params["adaptive_block_idx"]),
            "min_area": self.create_slider("min_area", "ขนาดเล็กสุด", 1, 5000, self.params["min_area"]),
            "max_area": self.create_slider("max_area", "ขนาดใหญ่สุด", 100, 20000, self.params["max_area"]),
            "erode_iters": self.create_slider("erode_iters", "ลด Noise (Erode)", 0, 5, self.params["erode_iters"]),
            "dilate_iters": self.create_slider("dilate_iters", "ขยายวัตถุ (Dilate)", 0, 5, self.params["dilate_iters"]),
            "kernel_size": self.create_slider("kernel_size", "ขนาด Kernel", 1, 7, self.params["kernel_size"])
        }
        for key, (slider, label) in self.sliders.items():
            slider.valueChanged.connect(self.update_param)
            layout.addWidget(label); layout.addWidget(slider)

        layout.addStretch()

        self.chk_auto_save = QCheckBox("บันทึกภาพอัตโนมัติเมื่อตรวจเจอ")
        self.chk_auto_save.clicked.connect(lambda: self.on_auto_save_changed())
        layout.addWidget(self.chk_auto_save)

        self.chk_play_sound = QCheckBox("ส่งเสียงเตือนเมื่อตรวจเจอ")
        self.chk_play_sound.clicked.connect(lambda: self.on_play_sound_changed())
        layout.addWidget(self.chk_play_sound)

        self.pause_button = QPushButton("หยุดชั่วคราว (P)"); self.pause_button.setCheckable(True); self.pause_button.clicked.connect(self.toggle_pause); layout.addWidget(self.pause_button)
        self.snapshot_button = QPushButton("บันทึกภาพนิ่ง (S)"); self.snapshot_button.clicked.connect(self.take_snapshot); layout.addWidget(self.snapshot_button)
        self.record_button = QPushButton("บันทึกวิดีโอ (V)"); self.record_button.setCheckable(True); self.record_button.clicked.connect(self.toggle_record); layout.addWidget(self.record_button)

        return layout

    def create_slider(self, key_name, display_name, min_val, max_val, initial_val):
        label_text = f"{display_name}: {2 * initial_val + 3}" if key_name == "adaptive_block_idx" else (
            f"{display_name}: {max(1, initial_val | 1)}" if key_name == "kernel_size" else f"{display_name}: {initial_val}")
        label = QLabel(label_text)
        slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(min_val, max_val); slider.setValue(initial_val); slider.setObjectName(key_name)
        return slider, label

    def create_video_panel(self):
        v = QVBoxLayout(); h = QHBoxLayout()
        self.video_label = QLabel("กำลังเชื่อมต่อกับแหล่งวิดีโอ…"); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setStyleSheet("background-color: black; color: white;")
        self.mask_label = QLabel("Mask"); self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.mask_label.setStyleSheet("background-color: black; color: white;")
        h.addWidget(self.video_label, 1); h.addWidget(self.mask_label, 1); v.addLayout(h)
        self.status_label = QLabel("สถานะ: กำลังเริ่มต้น…"); self.status_label.setFixedHeight(25); v.addWidget(self.status_label)
        return v

    def open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์วิดีโอ", "", "Video Files (*.mp4 *.m4v *.avi *.mov *.mkv);;All Files (*)")
        if not path: return
        self.switch_source("video", path)

    def switch_source(self, source_type: str, path: str | None):
        if source_type == "video":
            self.status_label.setText(f"สถานะ: เปิดไฟล์วิดีโอ -> {os.path.basename(path)}")
        else:
            self.status_label.setText("สถานะ: ใช้กล้อง (Webcam)")
        self.start_video_thread(source_type=source_type, video_path=path)

    def on_backend_changed(self, name: str):
        self.cam_backend_name = name

    def on_cam_index_changed(self, idx: int):
        self.cam_index = idx

    def on_fps_changed(self, val: int):
        self.target_fps = max(1, int(val))
        if self.video_worker is not None:
            self.video_worker.target_fps = self.target_fps

    def on_auto_save_changed(self):
        if self.video_worker:
            is_checked = self.chk_auto_save.isChecked()
            self.video_worker.auto_save_on_detect = is_checked
            print(f"[UI] Auto-save on detect set to: {is_checked}")
            self.chk_auto_save.update()

    def on_play_sound_changed(self):
        if self.video_worker:
            is_checked = self.chk_play_sound.isChecked()
            self.video_worker.play_sound_on_detect = is_checked
            print(f"[UI] Play sound on detect set to: {is_checked}")
            self.chk_play_sound.update()

    def update_param(self):
        sender = self.sender(); key = sender.objectName(); val = sender.value(); self.params[key] = val
        if self.video_worker is not None:
            self.video_worker.params[key] = val
        label = self.sliders[key][1]; name = label.text().split(':')[0]
        if key == "adaptive_block_idx": label.setText(f"{name}: {2 * val + 3}")
        elif key == "kernel_size": label.setText(f"{name}: {max(1, val | 1)}")
        else: label.setText(f"{name}: {val}")

    def update_gui_from_worker(self):
        vw = self.video_worker
        if vw is None: return
        with vw.lock:
            rf = vw.latest_result_frame.copy() if vw.latest_result_frame is not None else None
            mf = vw.latest_mask_frame.copy() if vw.latest_mask_frame is not None else None
            st = vw.latest_status_text
        if rf is not None:
            self.video_label.setPixmap(QPixmap.fromImage(self.to_qimage(rf)).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if mf is not None:
            self.mask_label.setPixmap(QPixmap.fromImage(self.to_qimage(mf, is_mask=True)).scaled(self.mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.status_label.setText(f"สถานะ: {st}")

    def to_qimage(self, img, is_mask=False):
        if is_mask:
            h, w = img.shape
            return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

    def toggle_pause(self, checked):
        if self.video_worker is None: return
        self.video_worker.paused = checked
        self.pause_button.setText("เล่นต่อ (P)" if checked else "หยุดชั่วคราว (P)")

    def take_snapshot(self):
        if self.video_worker is None: return
        self.video_worker.capture_snapshot = True

    def toggle_record(self, checked):
        vw = self.video_worker
        if vw is None or vw.latest_result_frame is None:
            self.record_button.setChecked(False); return
        h, w, _ = vw.latest_result_frame.shape
        is_rec, _ = vw.toggle_recording((w, h))
        self.record_button.setText("หยุดบันทึก (V)" if is_rec else "บันทึกวิดีโอ (V)")
        self.record_button.setStyleSheet("background-color: #C73E3A;" if is_rec else "")

    def show_connection_error(self, message):
        QMessageBox.critical(self, "Connection Error", f"{message}\n\nTips:\n• ลองสลับ Backend เป็น DSHOW หรือ MSMF\n• ตรวจสอบว่าไฟล์เข้ารหัสเป็น H.264 (.mp4)\n• ทดสอบเปิดด้วย VLC ได้ไหม")

    def closeEvent(self, event):
        self.save_settings()
        self.stop_video_thread()
        event.accept()

    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    self.params = settings.get("params", self.params)
                    self.cam_index = settings.get("cam_index", self.cam_index)
                    self.target_fps = settings.get("target_fps", self.target_fps)
                    self.cam_backend_name = settings.get("cam_backend_name", self.cam_backend_name)
                    print(f"[SETTINGS] Settings loaded from {SETTINGS_FILE}")
        except Exception as e:
            print(f"[ERROR] Could not load settings: {e}")

    def save_settings(self):
        try:
            settings = {
                "params": self.params,
                "cam_index": self.cam_index,
                "target_fps": self.target_fps,
                "cam_backend_name": self.cam_backend_name,
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"[SETTINGS] Settings saved to {SETTINGS_FILE}")
        except Exception as e:
            print(f"[ERROR] Could not save settings: {e}")

    def update_controls_from_settings(self):
        # Update sliders
        for key, (slider, label) in self.sliders.items():
            value = self.params.get(key, slider.value())
            slider.setValue(value)
            # Manually update label text
            name = label.text().split(':')[0]
            if key == "adaptive_block_idx": label.setText(f"{name}: {2 * value + 3}")
            elif key == "kernel_size": label.setText(f"{name}: {max(1, value | 1)}")
            else: label.setText(f"{name}: {value}")

        # Update other controls
        self.spin_cam_idx.setValue(self.cam_index)
        self.spin_fps.setValue(self.target_fps)
        self.cmb_backend.setCurrentText(self.cam_backend_name)


    def set_stylesheet(self):
        self.setStyleSheet("""
            QWidget { background-color: #2E2E2E; color: #E0E0E0; }
            QLabel { font-size: 14px; }
            QPushButton { background-color: #555; border: 1px solid #777; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #666; }
            QPushButton:checked { background-color: #007ACC; border: 1px solid #005C99; }
            QSlider::groove:horizontal { border: 1px solid #444; height: 8px; background: #333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #007ACC; border: 1px solid #005C99; width: 18px; margin: -5px 0; border-radius: 9px; }
            
            QCheckBox { 
                font-size: 14px; 
                spacing: 5px; 
                color: #E0E0E0;
            }
            QCheckBox::indicator {
                border: 1px solid #999;
                background-color: #555;
                width: 13px;
                height: 13px;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background-color: #007ACC;
                border: 1px solid #005C99;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #bbb;
            }
        """)

# ======================================================================
# Program Entry Point
# ======================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())
