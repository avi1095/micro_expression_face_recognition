import cv2
import mediapipe as mp
import libreface
import tempfile
import os
import torch
import threading
import queue
import csv
import numpy as np

mp_face_detection = mp.solutions.face_detection

class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def attribute_worker(input_queue, output_queue, device):
    while True:
        item = input_queue.get()
        if item is None:
            break
        frame_idx, face_path = item
        try:
            attrs = libreface.get_facial_attributes(face_path, device=device)
        except Exception as e:
            print(f"[Worker Error] Frame {frame_idx}: {e}")
            attrs = {}
        output_queue.put((frame_idx, attrs))
        input_queue.task_done()

def main():
    device = "cuda"
    print(f"Using device: {device}")

    stream = WebcamStream().start()
    mp_face = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    input_queue = queue.Queue(maxsize=10)
    output_queue = queue.Queue(maxsize=10)

    worker_thread = threading.Thread(target=attribute_worker, args=(input_queue, output_queue, device), daemon=True)
    worker_thread.start()

    csv_path = os.path.join('outputs', 'facial_attributes.csv')
    csv_writer = None
    csv_header = None

    frame_idx = 0
    display_attrs = {}
    last_frame = None

    with tempfile.TemporaryDirectory() as tmp_dir, open(csv_path, mode='w', newline='') as csvfile:
        try:
            while True:
                ret, frame = stream.read()
                if not ret:
                    if last_frame is None:
                        frame = 255 * np.ones((240, 320, 3), dtype=np.uint8)
                    else:
                        frame = last_frame
                    print("Warning: Failed to grab frame, using last or blank frame")
                else:
                    last_frame = frame

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face.process(image_rgb)

                if results.detections and frame_idx % 2 == 0:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1 = max(int(bbox.xmin * w), 0)
                        y1 = max(int(bbox.ymin * h), 0)
                        x2 = min(x1 + int(bbox.width * w), w - 1)
                        y2 = min(y1 + int(bbox.height * h), h - 1)

                        face_img = frame[y1:y2, x1:x2]

                        face_path = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
                        cv2.imwrite(face_path, face_img)

                        if not input_queue.full():
                            input_queue.put((frame_idx, face_path))

                while not output_queue.empty():
                    idx, attrs = output_queue.get()
                    display_attrs[idx] = attrs

                    if csv_writer is None and attrs:
                        csv_header = ['frame'] + list(attrs.keys())
                        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                        csv_writer.writeheader()

                    if csv_writer:
                        row = {'frame': idx}
                        if attrs:
                            row.update(attrs)
                        else:
                            zero_attrs = {key: 0 for key in csv_header if key != 'frame'}
                            row.update(zero_attrs)
                        csv_writer.writerow(row)

                if frame_idx in display_attrs and display_attrs[frame_idx]:
                    attrs = display_attrs[frame_idx]
                    if results.detections:
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = frame.shape
                            x1 = max(int(bbox.xmin * w), 0)
                            y1 = max(int(bbox.ymin * h), 0)
                            x2 = min(x1 + int(bbox.width * w), w - 1)
                            y2 = min(y1 + int(bbox.height * h), h - 1)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            attr_text = ', '.join(f"{k}: {v:.2f}" for k, v in attrs.items() if isinstance(v, (float, int)))
                            cv2.putText(frame, attr_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Webcam Facial Recognition", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                frame_idx += 1
        finally:
            stream.stop()
            mp_face.close()
            input_queue.put(None)
            worker_thread.join()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
