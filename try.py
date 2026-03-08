# # # import cv2
# # #
# # # video_path = "C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/gallery/video_me/Man.mp4"
# # # cap = cv2.VideoCapture(video_path)
# # # if not cap.isOpened():
# # #     print("OpenCV failed to open video.")
# # # else:
# # #     ret, frame = cap.read()
# # #     if ret:
# # #         print("Read a frame—showing it.")
# # #         cv2.imshow("Test Frame", frame)
# # #         cv2.waitKey(0)
# # #         cv2.destroyAllWindows()
# # #     else:
# # #         print("Failed to read a frame.")
# # # cap.release()
# # #
# # # # import os
# # # # print(os.environ["PATH"])
# # import cv2
# # import mediapipe as mp
# #
# # mp_face_detection = mp.solutions.face_detection
# #
# # # Set lower confidence thresholds to increase face detection sensitivity
# # with mp_face_detection.FaceDetection(
# #         min_detection_confidence=0.3,
# #         model_selection=0) as face_detection:
# #     cap = cv2.VideoCapture('c:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/gallery/video_me/Man.mp4')
# #     while cap.isOpened():
# #         success, frame = cap.read()
# #         if not success:
# #             break
# #
# #         # Convert frame to RGB for mediapipe
# #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = face_detection.process(image)
# #
# #         if results.detections:
# #             print(f"Detected {len(results.detections)} faces")
# #             # You can draw bounding boxes or landmarks here
# #         else:
# #             print("No faces detected")
# #
# #     cap.release()
# import cv2
# import mediapipe as mp
# import libreface
# import os
# import pandas as pd
#
# mp_face_detection = mp.solutions.face_detection
#
# video_path = r"C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/gallery/video_me/Man.mp4"
# output_dir = r"C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/outputs"
# os.makedirs(output_dir, exist_ok=True)
# output_csv_path = os.path.join(output_dir, "facial_attributes.csv")
#
# cap = cv2.VideoCapture(video_path)
# results = []
#
# with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
#     frame_idx = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detections = face_detection.process(image_rgb)
#
#         if detections.detections:
#             for detection in detections.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 h, w, _ = frame.shape
#                 x1 = max(int(bboxC.xmin * w), 0)
#                 y1 = max(int(bboxC.ymin * h), 0)
#                 x2 = min(x1 + int(bboxC.width * w), w - 1)
#                 y2 = min(y1 + int(bboxC.height * h), h - 1)
#
#                 face_img = frame[y1:y2, x1:x2]
#
#                 face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#
#                 # Run libreface on cropped face image (numpy array)
#                 attrs = libreface.get_facial_attributes(face_rgb, device="cpu")
#
#                 # Add frame and bbox info
#                 attrs["frame"] = frame_idx
#                 attrs["bbox"] = (x1, y1, x2, y2)
#
#                 results.append(attrs)
#
#         frame_idx += 1
#
# cap.release()
#
# df = pd.DataFrame(results)
# df.to_csv(output_csv_path, index=False)
#
# print(f"Facial recognition done! Results saved to: {output_csv_path}")
#
import torch
print(torch.__version__)         # Should print 2.0.0+cu118
print(torch.cuda.is_available()) # Should print True
print(torch.version.cuda)        # Should print '11.8'
