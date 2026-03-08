# # # import libreface
# # #
# # # video_path = "C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/gallery/video_me/newtry.avi"
# # # output_csv_path = "C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/outputs/my_video_result.csv"
# # #
# # # libreface.get_facial_attributes(video_path, output_csv_path, device="cpu")
# # import subprocess
# # try:
# #     subprocess.run(['ffmpeg', '-version'], check=True)
# #     print("ffmpeg is accessible from Python.")
# # except Exception as e:
# #     print("Python cannot find ffmpeg! Error:", e)
# import libreface
# import os
#
# video_path = "C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/gallery/video_me/newtry_fixed.mp4"
# output_csv_path = "C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/outputs/result.csv"
#
# libreface.get_facial_attributes(
#     video_path,
#     output_save_path=output_csv_path,
#     device="cpu"
# )
#
# print("Done? CSV exists:", os.path.exists(output_csv_path))
import os
video_path = "//gallery/video_me/man.mp4"
print("Man.mp4 exists?", os.path.exists(video_path))
print("File size:", os.path.getsize(video_path))
#_________________________
import libreface

video_path = "//gallery/video_me/Man.mp4"
output_csv_path = "//outputs/man_results.csv"

libreface.get_facial_attributes(
    video_path,
    output_save_path=output_csv_path,
    # device="cpu"
    device="cuda:0",  # use CPU if you prefer, but your RTX 4060 is excellent
    batch_size=64,
    num_workers=4
)

