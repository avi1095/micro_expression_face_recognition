import cv2
import mediapipe as mp
import libreface
import tempfile
import os
import torch
import threading

mp_face_detection = mp.solutions.face_detection

class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        # Set desired resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

def main():
    device = "cuda"  # Permanent GPU usage
    print(f"Using device: {device}")

    stream = WebcamStream().start()

    mp_face = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    frame_idx = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            while True:
                ret, frame = stream.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face.process(image_rgb)

                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1 = max(int(bbox.xmin * w), 0)
                        y1 = max(int(bbox.ymin * h), 0)
                        x2 = min(x1 + int(bbox.width * w), w - 1)
                        y2 = min(y1 + int(bbox.height * h), h - 1)

                        face_img = frame[y1:y2, x1:x2]

                        # Save cropped image temporarily and extract attributes on GPU
                        face_path = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
                        cv2.imwrite(face_path, face_img)

                        attrs = libreface.get_facial_attributes(face_path, device=device)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        attr_text = ', '.join(f"{k}: {v:.2f}" for k, v in attrs.items() if isinstance(v, (float, int)))
                        cv2.putText(frame, attr_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        print(f"Frame {frame_idx} attributes:", attrs)

                cv2.imshow("Webcam Facial Recognition", frame)

                if cv2.waitKey(1) & 0xFF == 27: 
                    break

                frame_idx += 1

        finally:
            stream.stop()
            mp_face.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
