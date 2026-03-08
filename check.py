# import libreface
# import os
#
# video_path = r"C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/gallery/video_me/perpvid.mp4"
# output_csv_path = r"C:/Users/ASUS/PycharmProjects/Micro Expression Face Recognition/outputs/perp_video_result.csv"
#
# print("Video file exists?", os.path.exists(video_path))
#
# libreface.get_facial_attributes(
#     video_path,
#     output_save_path=output_csv_path,
#     device="cpu",
#     batch_size=64,
#     num_workers=2
# )
#
# print("CSV file exists?", os.path.exists(output_csv_path))
import libreface
import os

video_path = r"/\gallery\video_me\newtry.avi"
output_csv_path = "//outputs/my_video_result_me.csv"

libreface.get_facial_attributes(
    video_path,
    output_save_path=output_csv_path,
    device="cuda:0",         # use CPU if you prefer, but your RTX 4060 is excellent
    batch_size=96,
    num_workers=6
)
print("CSV file exists?", os.path.exists(output_csv_path))
