import cv2
import numpy as np

def find_camera_index():
    """
    ฟังก์ชันสำหรับวนลูปค้นหาหมายเลข Index ของกล้องที่ใช้งานได้
    จะลองตั้งแต่ 0, 1, 2, ... ไปเรื่อยๆ
    """
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            print(f"พบกล้องที่ Index: {index}")
            cap.release()
            return index
        index += 1
    return -1 # คืนค่า -1 ถ้าไม่พบกล้องเลย

def main():
    """
    ฟังก์ชันหลักสำหรับเปิดกล้อง Webcam และตรวจจับแมลงแบบเรียลไทม์
    """
    # --- ปรับปรุง: ค้นหา Index ของกล้องโดยอัตโนมัติ ---
    camera_index = find_camera_index()
    
    if camera_index == -1:
        print("เกิดข้อผิดพลาด: ไม่พบกล้อง Webcam ในระบบ")
        return

    # เริ่มต้นการเชื่อมต่อกับ Webcam โดยใช้ Index ที่ค้นพบ
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"เกิดข้อผิดพลาด: ไม่สามารถเปิดกล้อง Webcam ที่ Index: {camera_index} ได้")
        return

    print("กล้องเปิดสำเร็จ กด 'q' เพื่อออกจากโปรแกรม")

    while True:
        # อ่านภาพจากกล้องทีละเฟรม
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถรับภาพจากกล้องได้ อาจมีปัญหาการเชื่อมต่อ")
            break

        # คัดลอกเฟรมภาพเพื่อวาดผลลัพธ์
        result_frame = frame.copy()

        # --- เริ่มต้นอัลกอริทึมการประมวลผลภาพ (เหมือนกับในเว็บแอป) ---

        # 1. แปลงภาพเป็น Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. ใช้ Dual Thresholding เพื่อหาจุดที่น่าจะเป็นแมลง
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
        _, global_thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        final_thresh = cv2.bitwise_and(adaptive_thresh, global_thresh)

        # 3. ทำ Morphological Operations เพื่อกำจัด Noise และทำรูปร่างให้สมบูรณ์
        kernel = np.ones((3, 3), np.uint8)
        final_thresh = cv2.erode(final_thresh, kernel, iterations=1)
        final_thresh = cv2.dilate(final_thresh, kernel, iterations=1)
        final_thresh = cv2.dilate(final_thresh, kernel, iterations=1)
        final_thresh = cv2.erode(final_thresh, kernel, iterations=1)

        # 4. ค้นหา Contours (เส้นขอบของวัตถุ)
        contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pest_count = 0
        # 5. วนลูปเพื่อกรองและวาดกรอบ
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if 18 < area < 1500:
                pest_count += 1
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 6. แสดงจำนวนที่ตรวจพบบนหน้าจอ
        cv2.putText(
            result_frame,
            f'Detected: {pest_count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA
        )
        
        # --- สิ้นสุดอัลกอริทึม ---

        # แสดงผลลัพธ์ในหน้าต่างชื่อ 'Real-time Detection'
        cv2.imshow('Real-time Detection', result_frame)
        
        # แสดงภาพ Mask ที่ผ่านการกรองแล้ว (สำหรับดีบัก)
        cv2.imshow('Processed Mask', final_thresh)

        # รอรับการกดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # เมื่อออกจากลูปแล้ว ให้ปล่อยการเชื่อมต่อกล้องและปิดหน้าต่างทั้งหมด
    cap.release()
    cv2.destroyAllWindows()
    print("โปรแกรมปิดการทำงานแล้ว")

if __name__ == '__main__':
    main()
