import cv2
import numpy as np
import sys

def find_camera():
    """
    ฟังก์ชันสำหรับค้นหากล้องที่ใช้งานได้ โดยจะลองทั้ง Backend DSHOW และ MSMF
    และคืนค่า Index กับ Backend ที่ดีที่สุด
    """
    print("กำลังค้นหากล้องที่เชื่อมต่อ...")
    found_cameras = []
    # วนลูปค้นหาทั้งสอง Backend ที่นิยมใช้บน Windows
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        for i in range(5): # ลองค้นหาสูงสุด 5 index
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"พบกล้อง: Index={i}, Backend={backend}, ความละเอียด: {int(width)}x{int(height)}")
                # เพิ่มเฉพาะกล้องที่ยังไม่เคยเจอเข้าไปในลิสต์
                if (i, backend) not in found_cameras:
                    found_cameras.append((i, backend))
                cap.release()
    
    if not found_cameras:
        return None, None
    
    # --- ปรับปรุง: คืนค่ากล้องตัวสุดท้ายที่เจอ ---
    # โดยทั่วไปกล้องที่เชื่อมต่อผ่าน USB ทีหลัง มักจะมี Index สูงกว่า
    print(f"เลือกใช้กล้องตัวสุดท้ายที่พบ: {found_cameras[-1]}")
    return found_cameras[-1]

def main():
    """
    ฟังก์ชันหลักสำหรับเปิดกล้อง Webcam, ตั้งค่าความละเอียด, และตรวจจับแมลง
    """
    camera_index, camera_backend = find_camera()

    if camera_index is None:
        print("เกิดข้อผิดพลาด: ไม่พบกล้อง Webcam ในระบบ")
        print("กรุณาตรวจสอบว่ากล้อง DJI Action 3 ได้เชื่อมต่อและอยู่ใน Webcam Mode แล้ว")
        sys.exit()

    print(f"กำลังเปิดกล้องที่ Index: {camera_index} ด้วย Backend: {camera_backend}")
    cap = cv2.VideoCapture(camera_index, camera_backend)

    # --- ตั้งค่าคุณสมบัติของกล้อง (ตามโค้ดที่คุณให้มา) ---
    # พยายามตั้งค่า Codec เป็น MJPG และความละเอียดเป็น 1080p ที่ 30fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"เกิดข้อผิดพลาด: ไม่สามารถเปิดกล้อง Webcam ที่ Index: {camera_index} ได้")
        sys.exit()

    print("กล้องเปิดสำเร็จ กด 'q' เพื่อออกจากโปรแกรม")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถรับภาพจากกล้องได้")
            break

        result_frame = frame.copy()

        # --- อัลกอริทึมการประมวลผลภาพ (เวอร์ชันที่ดีที่สุด) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
        _, global_thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        final_thresh = cv2.bitwise_and(adaptive_thresh, global_thresh)
        kernel = np.ones((3, 3), np.uint8)
        final_thresh = cv2.erode(final_thresh, kernel, iterations=1)
        final_thresh = cv2.dilate(final_thresh, kernel, iterations=1)
        final_thresh = cv2.dilate(final_thresh, kernel, iterations=1)
        final_thresh = cv2.erode(final_thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pest_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 18 < area < 1500:
                pest_count += 1
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            result_frame,
            f'Detected: {pest_count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA
        )
        
        # แสดงผลลัพธ์
        cv2.imshow('DJI Osmo Action 3 - Pest Detection', result_frame)
        cv2.imshow('Processed Mask', final_thresh)

        # รอรับการกดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปิดการเชื่อมต่อและหน้าต่างทั้งหมด
    cap.release()
    cv2.destroyAllWindows()
    print("โปรแกรมปิดการทำงานแล้ว")

if __name__ == '__main__':
    main()
