import sys
import os
import json
import time
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QMessageBox, QFileDialog, QComboBox, QSpinBox, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QPoint, QEvent, QRect

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
SOUND_FILE = "alert.wav"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================================
# Video Processing Worker Thread
# ======================================================================
class VideoWorker(QObject):
    connection_failed = pyqtSignal(str)
    playSound = pyqtSignal()

    def __init__(self, source_type="camera", video_path=None, params=None,
                 cam_index=0, target_fps=30,
                 desired_width=1920, desired_height=1080,
                 session_save_path="captures",
                 session_timestamp=""):
        super().__init__()
        self.running = True
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.params = params or {
            "global_thresh": 125, "adaptive_c": 2, "adaptive_block_idx": 9,
            "min_area": 18, "max_area": 1500, "erode_iters": 1,
            "dilate_iters": 2, "kernel_size": 3
        }
        self.lock = threading.Lock()
        self.latest_result_frame = None
        self.latest_mask_frame = None
        self.latest_original_frame = None
        self.latest_status_text = "Initializing..."

        self.source_type = source_type
        self.video_path = video_path
        self.cam_index = cam_index
        self.target_fps = target_fps
        self.desired_width = desired_width
        self.desired_height = desired_height

        self.detection_enabled = True
        self.auto_save_on_detect = False
        self.save_original_enabled = True
        self.play_sound_on_detect = False
        self.last_event_time = 0
        self.event_cooldown_seconds = 3.0 
        
        self.last_sound_time = 0
        self.sound_cooldown_seconds = 3.0

        self.roi_enabled = False
        self.roi_points = []

        self.session_save_path = session_save_path
        self.session_timestamp = session_timestamp
        self.detection_log = []

    def run(self):
        cap = None
        try:
            if self.source_type == "image":
                if not self.video_path or not os.path.exists(self.video_path):
                    self.connection_failed.emit("ไม่พบไฟล์ภาพหรือพาธไม่ถูกต้อง")
                    return
                
                frame = cv2.imread(self.video_path)
                if frame is None:
                    self.connection_failed.emit(f"ไม่สามารถอ่านไฟล์ภาพได้:\n{self.video_path}")
                    return
                
                print(f"[SOURCE] Using image file: {self.video_path}")
                
                if self.detection_enabled:
                    result_frame, mask_frame, pest_count, mean_area = self.process_frame(frame)
                    status_text = f"Detected: {pest_count} | Mean Area: {mean_area:.0f} | File: {os.path.basename(self.video_path)}"
                    if pest_count > 0:
                        ts_for_log = datetime.now()
                        self.detection_log.append((ts_for_log, pest_count))
                        if self.play_sound_on_detect: self.playSound.emit()
                        if self.auto_save_on_detect:
                            ts_str = ts_for_log.strftime('%Y%m%d_%H%M%S_%f')[:-3]
                            detected_filename = os.path.join(self.session_save_path, f"auto_detect_{ts_str}_detected.png")
                            cv2.imwrite(detected_filename, result_frame)
                            if self.save_original_enabled:
                                original_filename = os.path.join(self.session_save_path, f"auto_detect_{ts_str}_original.png")
                                cv2.imwrite(original_filename, frame)
                            print(f"[AUTO-SAVE] Image(s) saved to {self.session_save_path}")
                else:
                    result_frame = frame.copy()
                    mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    status_text = "การตรวจจับปิดอยู่"

                with self.lock:
                    self.latest_original_frame = frame.copy()
                    self.latest_result_frame = result_frame
                    self.latest_mask_frame = mask_frame
                    self.latest_status_text = status_text
                return

            if self.source_type == "video":
                if not self.video_path or not os.path.exists(self.video_path):
                    self.connection_failed.emit("ไม่พบไฟล์วิดีโอหรือพาธไม่ถูกต้อง")
                    return
                cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.connection_failed.emit(f"เปิดไฟล์วิดีโอไม่สำเร็จ (FFMPEG):\n{self.video_path}\n\nแนะนำเข้ารหัสเป็น H.264 (.mp4) ถ้ายังเปิดไม่ได้")
                    return
                self.print_video_props(cap)
                print(f"[SOURCE] Using video file (FFMPEG): {self.video_path}")
            else: # Camera
                print(f"[SOURCE] Auto-detecting backend for camera index {self.cam_index}...")
                backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
                for be in backends_to_try:
                    be_name = "Default" if be is None else ("DSHOW" if be == cv2.CAP_DSHOW else "MSMF")
                    print(f"[SOURCE]   -> Trying backend: {be_name}")
                    cap = cv2.VideoCapture(self.cam_index) if be is None else cv2.VideoCapture(self.cam_index, be)
                    if cap and cap.isOpened():
                        print(f"[SOURCE] Successfully opened camera with backend: {be_name}")
                        break
                    else:
                        if cap: cap.release()
                        cap = None
                if cap is None:
                    self.connection_failed.emit(f"ไม่พบหรือเปิดกล้องไม่ได้ (index={self.cam_index})\nกรุณาตรวจสอบไดรเวอร์กล้อง")
                    return
                
                print(f"[SOURCE] Using camera index {self.cam_index}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if self.desired_width > 0 and self.desired_height > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.desired_width))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.desired_height))
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

                if self.detection_enabled:
                    result_frame, mask_frame, pest_count, mean_area = self.process_frame(frame)
                    is_detecting_now = pest_count > 0

                    if is_detecting_now:
                        current_time = time.time()
                        if current_time - self.last_event_time > self.event_cooldown_seconds:
                            ts_for_log = datetime.now()
                            self.detection_log.append((ts_for_log, pest_count))
                            if self.auto_save_on_detect:
                                ts_str = ts_for_log.strftime('%Y%m%d_%H%M%S_%f')[:-3]
                                detected_filename = os.path.join(self.session_save_path, f"auto_detect_{ts_str}_detected.png")
                                cv2.imwrite(detected_filename, result_frame)
                                if self.save_original_enabled:
                                    original_filename = os.path.join(self.session_save_path, f"auto_detect_{ts_str}_original.png")
                                    cv2.imwrite(original_filename, frame)
                                print(f"[AUTO-SAVE] Image(s) saved to {self.session_save_path}")
                            self.last_event_time = current_time
                    
                    if is_detecting_now and self.play_sound_on_detect:
                        current_time = time.time()
                        if current_time - self.last_sound_time > self.sound_cooldown_seconds:
                            self.playSound.emit()
                            self.last_sound_time = current_time
                    
                    if not is_detecting_now:
                        self.last_sound_time = 0
                    
                    now = time.time()
                    fps = 1.0 / max(1e-6, (now - last_t)); last_t = now; n += 1
                    fps_avg = (fps_avg * (n - 1) + fps) / n
                    status_text = (f"Detected: {pest_count} | FPS: {fps:.1f} (avg {fps_avg:.1f}) | Mean Area: {mean_area:.0f}")
                else:
                    result_frame = frame.copy()
                    mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    status_text = "การตรวจจับปิดอยู่"

                if self.recording and self.video_writer is not None:
                    self.video_writer.write(result_frame)

                with self.lock:
                    self.latest_original_frame = frame.copy()
                    self.latest_result_frame = result_frame
                    self.latest_mask_frame = mask_frame
                    self.latest_status_text = status_text

                elapsed = time.time() - start_t
                if frame_interval > 0 and frame_interval > elapsed:
                    time.sleep(frame_interval - elapsed)
        finally:
            if cap is not None:
                cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
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
        
        offset_x, offset_y = 0, 0
        processing_gray = gray

        if self.roi_enabled and self.roi_points:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            offset_x, offset_y = x1, y1
            
            processing_gray = gray[y1:y2, x1:x2]
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if processing_gray.shape[0] == 0 or processing_gray.shape[1] == 0:
            return result_frame, np.zeros(gray.shape, dtype=np.uint8), 0, 0

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
        
        adaptive = cv2.adaptiveThreshold(processing_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block, C)
        _, global_t = cv2.threshold(processing_gray, gT, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(adaptive, global_t)
        kernel = np.ones((ksz, ksz), np.uint8)
        
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
                cv2.rectangle(result_frame, (x + offset_x, y + offset_y), (x + w + offset_x, y + h + offset_y), (0, 255, 0), 2)
                
        mean_area = (sum(areas) / len(areas)) if areas else 0
        
        final_display_mask = np.zeros(gray.shape, dtype=np.uint8)
        if self.roi_enabled and self.roi_points:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            if mask.shape[0] == (y2 - y1) and mask.shape[1] == (x2 - x1):
                final_display_mask[y1:y2, x1:x2] = mask
        else:
            final_display_mask = mask

        return result_frame, final_display_mask, count, mean_area

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

class ROISelectorLabel(QLabel):
    roi_selected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_selecting = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        # Crosshair settings
        self.crosshair_enabled = True
        self.crosshair_size = 20
        self.crosshair_thickness = 2

    def start_selection(self):
        self.is_selecting = True
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if self.is_selecting and event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_selecting and event.buttons() == Qt.MouseButton.LeftButton:
            self.end_point = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_selecting and event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.roi_selected.emit(selection_rect)
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        # ROI selection rectangle
        if self.is_selecting:
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())
        # Centered crosshair
        if self.crosshair_enabled:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(Qt.GlobalColor.red, self.crosshair_thickness)
            painter.setPen(pen)
            cx = self.width() // 2
            cy = self.height() // 2
            sz = self.crosshair_size
            painter.drawLine(cx - sz, cy, cx + sz, cy)
            painter.drawLine(cx, cy - sz, cx, cy + sz)
            painter.drawPoint(cx, cy)


# ======================================================================
# Main Application Window (UI)
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("โปรแกรมตรวจจับแมลง – Camera/Video")
        self.setGeometry(100, 100, 1700, 920)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.params = {
            "global_thresh": 125, "adaptive_c": 2, "adaptive_block_idx": 9,
            "min_area": 18, "max_area": 1500, "erode_iters": 1,
            "dilate_iters": 2, "kernel_size": 3
        }
        self.cam_index = 0
        self.target_fps = 30
        self.pref_width = 1920
        self.pref_height = 1080
        
        self.video_thread = None
        self.video_worker = None

        self.current_source_type = "camera"
        self.current_source_path = None
        self.session_timestamp = None
        self.session_save_path = None
        
        self.pref_detection_enabled = True
        self.pref_auto_save = False
        self.pref_play_sound = False
        self.pref_save_original = True
        
        self.pref_roi_enabled = False
        self.pref_roi_points = []

        self.load_settings()

        controls_layout = self.create_controls_panel()
        main_layout.addLayout(controls_layout, 1)
        video_layout = self.create_video_panel()
        main_layout.addLayout(video_layout, 4)

        self.update_controls_from_settings()

        self.switch_source("camera", None)

        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_gui_from_worker)
        self.ui_update_timer.start(33)
        self.set_stylesheet()

    def start_video_thread(self, source_type: str, video_path: str | None):
        self.stop_video_thread()

        self.video_worker = VideoWorker(
            source_type=source_type,
            video_path=video_path,
            params=self.params.copy(),
            cam_index=self.cam_index,
            target_fps=self.target_fps,
            desired_width=self.pref_width,
            desired_height=self.pref_height,
            session_save_path=self.session_save_path,
            session_timestamp=self.session_timestamp,
        )
        if hasattr(self, 'chk_enable_detection'):
            self.video_worker.detection_enabled = self.chk_enable_detection.isChecked()
            self.video_worker.auto_save_on_detect = self.chk_auto_save.isChecked()
            self.video_worker.save_original_enabled = self.chk_save_original.isChecked()
            self.video_worker.play_sound_on_detect = self.chk_play_sound.isChecked()
            self.video_worker.roi_enabled = self.chk_enable_roi.isChecked()
            if self.pref_roi_points:
                self.video_worker.roi_points = [
                    (self.pref_roi_points[0], self.pref_roi_points[1]),
                    (self.pref_roi_points[2], self.pref_roi_points[3])
                ]
        else:
            self.video_worker.detection_enabled = self.pref_detection_enabled
            self.video_worker.auto_save_on_detect = self.pref_auto_save
            self.video_worker.save_original_enabled = self.pref_save_original
            self.video_worker.play_sound_on_detect = self.pref_play_sound
            self.video_worker.roi_enabled = self.pref_roi_enabled
            if self.pref_roi_points:
                self.video_worker.roi_points = [
                    (self.pref_roi_points[0], self.pref_roi_points[1]),
                    (self.pref_roi_points[2], self.pref_roi_points[3])
                ]

        self.video_thread = QThread()
        self.video_worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.connection_failed.connect(self.show_connection_error)
        self.video_worker.playSound.connect(self.play_alert_sound)
        self.video_thread.start()

    def stop_video_thread(self):
        if self.video_thread is not None and self.video_worker is not None:
            self.video_worker.running = False
            self.video_thread.quit(); self.video_thread.wait()
            self.video_worker.deleteLater(); self.video_thread.deleteLater()
        self.video_worker = None; self.video_thread = None

    def create_controls_panel(self):
        layout = QVBoxLayout(); layout.setSpacing(10); layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Cam Index"))
        self.spin_cam_idx = QSpinBox(); self.spin_cam_idx.setRange(0, 10)
        self.spin_cam_idx.valueChanged.connect(self.on_cam_index_changed)
        cam_row.addWidget(self.spin_cam_idx)
        layout.addLayout(cam_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Target FPS"))
        self.spin_fps = QSpinBox(); self.spin_fps.setRange(1, 120)
        self.spin_fps.valueChanged.connect(self.on_fps_changed)
        fps_row.addWidget(self.spin_fps)
        layout.addLayout(fps_row)

        # Resolution controls
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution (W×H)"))
        self.spin_width = QSpinBox(); self.spin_width.setRange(160, 7680); self.spin_width.setSingleStep(16)
        self.spin_height = QSpinBox(); self.spin_height.setRange(120, 4320); self.spin_height.setSingleStep(16)
        self.spin_width.valueChanged.connect(self.on_resolution_changed)
        self.spin_height.valueChanged.connect(self.on_resolution_changed)
        res_row.addWidget(self.spin_width)
        res_row.addWidget(QLabel("×"))
        res_row.addWidget(self.spin_height)
        layout.addLayout(res_row)

        self.btn_open_source = QPushButton("เลือกไฟล์ภาพ/วิดีโอ…")
        self.btn_open_source.clicked.connect(self.open_source_file)
        layout.addWidget(self.btn_open_source)
        
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
        
        self.chk_enable_roi = QCheckBox("เปิดใช้งาน ROI")
        self.chk_enable_roi.clicked.connect(self.on_enable_roi_changed)
        layout.addWidget(self.chk_enable_roi)

        self.btn_set_roi = QPushButton("กำหนดพื้นที่ (ROI)")
        self.btn_set_roi.clicked.connect(self.start_roi_selection)
        layout.addWidget(self.btn_set_roi)

        self.chk_enable_detection = QCheckBox("เปิดใช้งานการตรวจจับ")
        self.chk_enable_detection.clicked.connect(self.on_enable_detection_changed)
        layout.addWidget(self.chk_enable_detection)

        self.chk_auto_save = QCheckBox("บันทึกภาพอัตโนมัติเมื่อตรวจเจอ")
        self.chk_auto_save.clicked.connect(lambda: self.on_auto_save_changed())
        layout.addWidget(self.chk_auto_save)
        
        self.chk_save_original = QCheckBox("บันทึกภาพต้นฉบับด้วย (เมื่อ Auto-Save)")
        self.chk_save_original.clicked.connect(self.on_save_original_changed)
        layout.addWidget(self.chk_save_original)

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
        self.video_label = ROISelectorLabel("กำลังเชื่อมต่อกับแหล่งวิดีโอ…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.roi_selected.connect(self.on_roi_selected)
        
        self.mask_label = QLabel("Mask"); self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.mask_label.setStyleSheet("background-color: black; color: white;")
        h.addWidget(self.video_label, 1); h.addWidget(self.mask_label, 1); v.addLayout(h)
        self.status_label = QLabel("สถานะ: กำลังเริ่มต้น…"); self.status_label.setFixedHeight(25); v.addWidget(self.status_label)

        # --- NEW: current resolution label ---
        self.resolution_label = QLabel("ความละเอียดปัจจุบัน: -")
        self.resolution_label.setFixedHeight(24)
        v.addWidget(self.resolution_label)

        return v

    def open_source_file(self):
        image_extensions = "*.jpg *.jpeg *.png *.bmp"
        video_extensions = "*.mp4 *.m4v *.avi *.mov *.mkv"
        dialog_filter = f"All Media Files ({image_extensions} {video_extensions});;Image Files ({image_extensions});;Video Files ({video_extensions})"
        
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์ภาพหรือวิดีโอ", "", dialog_filter)
        
        if not path:
            return
            
        _, ext = os.path.splitext(path)
        if ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            source_type = "image"
        else:
            source_type = "video"
            
        self.switch_source(source_type, path)

    def switch_source(self, source_type: str, path: str | None):
        self.current_source_type = source_type
        self.current_source_path = path

        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_save_path = os.path.join(SAVE_DIR, self.session_timestamp)
        os.makedirs(self.session_save_path, exist_ok=True)
        print(f"[SESSION] New session started. Saving files to: {self.session_save_path}")

        if source_type == "video":
            self.status_label.setText(f"สถานะ: เปิดไฟล์วิดีโอ -> {os.path.basename(path)}")
        elif source_type == "image":
            self.status_label.setText(f"สถานะ: เปิดไฟล์ภาพ -> {os.path.basename(path)}")
        else: # camera
            self.status_label.setText("สถานะ: ใช้กล้อง (Webcam)")
            
        self.start_video_thread(source_type=source_type, video_path=path)

    def on_cam_index_changed(self, idx: int):
        self.cam_index = idx

    def on_fps_changed(self, val: int):
        self.target_fps = max(1, int(val))
        if self.video_worker is not None:
            self.video_worker.target_fps = self.target_fps

    def on_resolution_changed(self):
        self.pref_width = int(self.spin_width.value())
        self.pref_height = int(self.spin_height.value())
        self.save_settings()
        if self.current_source_type == 'camera':
            self.start_video_thread(self.current_source_type, self.current_source_path)

    def on_auto_save_changed(self):
        if self.video_worker:
            is_checked = self.chk_auto_save.isChecked()
            self.video_worker.auto_save_on_detect = is_checked
            print(f"[UI] Auto-save on detect set to: {is_checked}")
        self.pref_auto_save = self.chk_auto_save.isChecked()
        self.save_settings()
        self.chk_auto_save.update()

    def on_save_original_changed(self):
        if self.video_worker:
            is_checked = self.chk_save_original.isChecked()
            self.video_worker.save_original_enabled = is_checked
            print(f"[UI] Save original on auto-save set to: {is_checked}")
        self.pref_save_original = self.chk_save_original.isChecked()
        self.save_settings()
        self.chk_save_original.update()

    def on_play_sound_changed(self):
        if self.video_worker:
            is_checked = self.chk_play_sound.isChecked()
            self.video_worker.play_sound_on_detect = is_checked
            print(f"[UI] Play sound on detect set to: {is_checked}")
        self.pref_play_sound = self.chk_play_sound.isChecked()
        self.save_settings()
        self.chk_play_sound.update()
        
    def on_enable_detection_changed(self):
        if self.video_worker:
            is_checked = self.chk_enable_detection.isChecked()
            self.video_worker.detection_enabled = is_checked
            print(f"[UI] Detection enabled set to: {is_checked}")
        self.pref_detection_enabled = self.chk_enable_detection.isChecked()
        self.save_settings()
        self.chk_enable_detection.update()

    def on_enable_roi_changed(self):
        if self.video_worker:
            is_checked = self.chk_enable_roi.isChecked()
            self.video_worker.roi_enabled = is_checked
            print(f"[UI] ROI enabled set to: {is_checked}")
        self.pref_roi_enabled = self.chk_enable_roi.isChecked()
        self.save_settings()
        self.chk_enable_roi.update()
        if self.current_source_type == 'image':
            self.start_video_thread(self.current_source_type, self.current_source_path)

    def start_roi_selection(self):
        self.video_label.start_selection()
        self.status_label.setText("สถานะ: คลิกและลากเพื่อกำหนดพื้นที่ (ROI)")

    def play_alert_sound(self):
        if winsound and os.path.exists(SOUND_FILE):
            try:
                winsound.PlaySound(SOUND_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print(f"[ERROR] Could not play sound file '{SOUND_FILE}': {e}. Trying fallback.")
                print('\a', flush=True)
        else:
            if not os.path.exists(SOUND_FILE):
                print(f"[WARNING] Sound file not found: {SOUND_FILE}. Using fallback beep.")
            print('\a', flush=True)

    def update_param(self):
        sender = self.sender(); key = sender.objectName(); val = sender.value(); self.params[key] = val
        if self.video_worker is not None:
            self.video_worker.params[key] = val
        label = self.sliders[key][1]; name = label.text().split(':')[0]
        if key == "adaptive_block_idx": label.setText(f"{name}: {2 * val + 3}")
        elif key == "kernel_size": label.setText(f"{name}: {max(1, val | 1)}")
        else: label.setText(f"{name}: {val}")

        if self.current_source_type == 'image':
            print("[UI] Parameter changed for image, reprocessing...")
            self.start_video_thread(self.current_source_type, self.current_source_path)

    def update_gui_from_worker(self):
        vw = self.video_worker
        if vw is None: return
        with vw.lock:
            rf = vw.latest_result_frame.copy() if vw.latest_result_frame is not None else None
            mf = vw.latest_mask_frame.copy() if vw.latest_mask_frame is not None else None
            st = vw.latest_status_text
        if rf is not None:
            self.video_label.setPixmap(QPixmap.fromImage(self.to_qimage(rf)).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            # --- NEW: show current resolution from latest frame ---
            h, w = rf.shape[:2]
            self.resolution_label.setText(f"ความละเอียดปัจจุบัน: {w}×{h}")
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
        if self.video_worker is None:
            self.status_label.setText("สถานะ: ไม่สามารถบันทึกภาพได้ (Worker ไม่พร้อม)")
            return

        original_frame = None
        result_frame = None
        session_path = self.session_save_path
        
        with self.video_worker.lock:
            if self.video_worker.latest_result_frame is not None:
                result_frame = self.video_worker.latest_result_frame.copy()
                original_frame = self.video_worker.latest_original_frame.copy()
        
        if result_frame is not None and original_frame is not None and session_path is not None:
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                detected_path = os.path.join(session_path, f"snap_{ts}_detected.png")
                cv2.imwrite(detected_path, result_frame)
                original_path = os.path.join(session_path, f"snap_{ts}_original.png")
                cv2.imwrite(original_path, original_frame)
                self.status_label.setText(f"สถานะ: บันทึกภาพนิ่งแล้ว")
                print(f"[UI] Snapshot saved to {session_path}")
            except Exception as e:
                self.status_label.setText(f"สถานะ: บันทึกภาพนิ่งล้มเหลว")
                print(f"[ERROR] Failed to save snapshot: {e}")
        else:
            self.status_label.setText(f"สถานะ: ไม่มีภาพให้บันทึก")

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
                    self.pref_width = settings.get("camera_width", getattr(self, 'pref_width', 1920))
                    self.pref_height = settings.get("camera_height", getattr(self, 'pref_height', 1080))
                    self.pref_detection_enabled = settings.get("detection_enabled", True)
                    self.pref_auto_save = settings.get("auto_save", False)
                    self.pref_save_original = settings.get("save_original", True)
                    self.pref_play_sound = settings.get("play_sound", False)
                    self.pref_roi_enabled = settings.get("roi_enabled", False)
                    self.pref_roi_points = settings.get("roi_points", [])
                    print(f"[SETTINGS] Settings loaded from {SETTINGS_FILE}")
        except Exception as e:
            print(f"[ERROR] Could not load settings: {e}")

    def save_settings(self):
        try:
            settings = {
                "params": self.params,
                "cam_index": self.cam_index,
                "target_fps": self.target_fps,
                "camera_width": getattr(self, 'pref_width', 1920),
                "camera_height": getattr(self, 'pref_height', 1080),
                "detection_enabled": self.pref_detection_enabled,
                "auto_save": self.pref_auto_save,
                "save_original": self.pref_save_original,
                "play_sound": self.pref_play_sound,
                "roi_enabled": self.pref_roi_enabled,
                "roi_points": self.pref_roi_points,
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"[SETTINGS] Settings saved to {SETTINGS_FILE}")
        except Exception as e:
            print(f"[ERROR] Could not save settings: {e}")

    def update_controls_from_settings(self):
        for key, (slider, label) in self.sliders.items():
            value = self.params.get(key, slider.value())
            slider.setValue(value)
            name = label.text().split(':')[0]
            if key == "adaptive_block_idx": label.setText(f"{name}: {2 * value + 3}")
            elif key == "kernel_size": label.setText(f"{name}: {max(1, value | 1)}")
            else: label.setText(f"{name}: {value}")

        self.spin_cam_idx.setValue(self.cam_index)
        self.spin_fps.setValue(self.target_fps)
        if hasattr(self, 'spin_width'):
            self.spin_width.setValue(getattr(self, 'pref_width', 1920))
        if hasattr(self, 'spin_height'):
            self.spin_height.setValue(getattr(self, 'pref_height', 1080))
        
        self.chk_enable_detection.setChecked(self.pref_detection_enabled)
        self.chk_auto_save.setChecked(self.pref_auto_save)
        self.chk_save_original.setChecked(self.pref_save_original)
        self.chk_play_sound.setChecked(self.pref_play_sound)
        self.chk_enable_roi.setChecked(self.pref_roi_enabled)

    def on_roi_selected(self, selection_rect_in_label):
        if self.video_worker is None: return
        with self.video_worker.lock:
            if self.video_worker.latest_original_frame is None:
                return
            original_frame_size = self.video_worker.latest_original_frame.shape
        
        h_orig, w_orig = original_frame_size[0], original_frame_size[1]
        label_size = self.video_label.size()
        pixmap = self.video_label.pixmap()
        if not pixmap or pixmap.isNull():
            return
        
        scaled_pixmap_size = pixmap.size().scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)
        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2
        
        if scaled_pixmap_size.width() == 0 or scaled_pixmap_size.height() == 0:
            return

        scale_x = w_orig / scaled_pixmap_size.width()
        scale_y = h_orig / scaled_pixmap_size.height()

        x1 = int((selection_rect_in_label.left() - offset_x) * scale_x)
        y1 = int((selection_rect_in_label.top() - offset_y) * scale_y)
        x2 = int((selection_rect_in_label.right() - offset_x) * scale_x)
        y2 = int((selection_rect_in_label.bottom() - offset_y) * scale_y)

        final_x1 = max(0, x1)
        final_y1 = max(0, y1)
        final_x2 = min(w_orig, x2)
        final_y2 = min(h_orig, y2)
        
        self.pref_roi_points = [final_x1, final_y1, final_x2, final_y2]
        if self.video_worker:
            self.video_worker.roi_points = [(final_x1, final_y1), (final_x2, final_y2)]
        self.save_settings()
        
        print(f"[UI] ROI set to: {self.pref_roi_points}")
        self.status_label.setText("สถานะ: กำหนดพื้นที่เรียบร้อยแล้ว")
        
        if self.current_source_type == 'image':
            self.start_video_thread(self.current_source_type, self.current_source_path)


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
