# import cv2
# import mediapipe as mp
# import libreface
# import tempfile
# import os
# import torch
# import threading
# import csv
#
# mp_face_detection = mp.solutions.face_detection
#
# class WebcamStream:
#     def __init__(self, src=0):
#         self.cap = cv2.VideoCapture(src)
#         # Set desired resolution and FPS
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         self.cap.set(cv2.CAP_PROP_FPS, 15)
#
#         self.ret, self.frame = self.cap.read()
#         self.stopped = False
#         self.lock = threading.Lock()
#
#     def start(self):
#         threading.Thread(target=self.update, daemon=True).start()
#         return self
#
#     def update(self):
#         while not self.stopped:
#             ret, frame = self.cap.read()
#             with self.lock:
#                 self.ret = ret
#                 self.frame = frame
#
#     def read(self):
#         with self.lock:
#             return self.ret, self.frame
#
#     def stop(self):
#         self.stopped = True
#         self.cap.release()
#
# def main():
#     device = "cuda"  # Permanent GPU usage
#     print(f"Using device: {device}")
#
#     stream = WebcamStream().start()
#     mp_face = mp_face_detection.FaceDetection(min_detection_confidence=0.3)
#
#     frame_idx = 0
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         # Open CSV file in write mode
#         # with open('facial_attributes.csv', mode='w', newline='') as csvfile:
#         csv_path = os.path.join('outputs', 'facial_attributes.csv')
#         with open(csv_path, mode='w', newline='') as csvfile:
#
#             csv_writer = None
#
#             try:
#                 while True:
#                     ret, frame = stream.read()
#                     if not ret:
#                         print("Failed to grab frame")
#                         break
#
#                     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     results = mp_face.process(image_rgb)
#
#                     if results.detections:
#                         for detection in results.detections:
#                             bbox = detection.location_data.relative_bounding_box
#                             h, w, _ = frame.shape
#                             x1 = max(int(bbox.xmin * w), 0)
#                             y1 = max(int(bbox.ymin * h), 0)
#                             x2 = min(x1 + int(bbox.width * w), w - 1)
#                             y2 = min(y1 + int(bbox.height * h), h - 1)
#
#                             face_img = frame[y1:y2, x1:x2]
#
#                             face_path = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
#                             cv2.imwrite(face_path, face_img)
#
#                             attrs = libreface.get_facial_attributes(face_path, device=device)
#
#                             # Initialize CSV writer with header based on attribute keys
#                             if csv_writer is None:
#                                 header = ['frame'] + list(attrs.keys())
#                                 csv_writer = csv.DictWriter(csvfile, fieldnames=header)
#                                 csv_writer.writeheader()
#
#                             # Write row: frame number + attribute values
#                             row = {'frame': frame_idx}
#                             row.update(attrs)
#                             csv_writer.writerow(row)
#
#                             # Display bounding box and attribute text on frame
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             attr_text = ', '.join(f"{k}: {v:.2f}" for k, v in attrs.items() if isinstance(v, (float, int)))
#                             cv2.putText(frame, attr_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#
#                             print(f"Frame {frame_idx} attributes:", attrs)
#
#                     cv2.imshow("Webcam Facial Recognition", frame)
#
#                     if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
#                         break
#
#                     frame_idx += 1
#
#             finally:
#                 stream.stop()
#                 mp_face.close()
#                 cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
import cv2
import mediapipe as mp
import libreface
import tempfile
import os
import torch
import threading
import csv
import numpy as np  # For blank frame creation if needed

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
    last_frame = None
    csv_writer = None
    csv_header = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join('outputs', 'facial_attributes.csv')
        with open(csv_path, mode='w', newline='') as csvfile:
            try:
                while True:
                    ret, frame = stream.read()
                    if not ret:
                        print("Warning: Failed to grab frame, using last frame or blank")
                        if last_frame is None:
                            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)  # white blank frame
                        else:
                            frame = last_frame
                    else:
                        last_frame = frame

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

                            face_path = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
                            cv2.imwrite(face_path, face_img)

                            try:
                                attrs = libreface.get_facial_attributes(face_path, device=device)
                            except RuntimeError as e:
                                if "No face landmarks" in str(e):
                                    print(f"Frame {frame_idx}: No face landmarks found, recording zero AUs")
                                    attrs = {}
                                else:
                                    raise

                            if csv_writer is None and attrs:
                                csv_header = ['frame'] + list(attrs.keys())
                                csv_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                                csv_writer.writeheader()

                            if attrs:
                                row = {'frame': frame_idx}
                                row.update(attrs)
                                csv_writer.writerow(row)

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                attr_text = ', '.join(f"{k}: {v:.2f}" for k, v in attrs.items() if isinstance(v, (float, int)))
                                cv2.putText(frame, attr_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                print(f"Frame {frame_idx} attributes:", attrs)

                            else:
                                # Write zero AUs row if attributes empty
                                if csv_writer:
                                    row = {'frame': frame_idx}
                                    zero_attrs = {key: 0 for key in csv_header if key != 'frame'}
                                    row.update(zero_attrs)
                                    csv_writer.writerow(row)
                    else:
                        print(f"Frame {frame_idx}: No face detected")

                        if csv_writer:
                            row = {'frame': frame_idx}
                            zero_attrs = {key: 0 for key in csv_header if key != 'frame'}
                            row.update(zero_attrs)
                            csv_writer.writerow(row)

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
