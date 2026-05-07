import os
video_path = "//gallery/video_me/man.mp4"
print("Man.mp4 exists?", os.path.exists(video_path))
print("File size:", os.path.getsize(video_path))
import libreface

video_path = "//gallery/video_me/Man.mp4"
output_csv_path = "//outputs/man_results.csv"

libreface.get_facial_attributes(
    video_path,
    output_save_path=output_csv_path,
    device="cuda:0",
    batch_size=64,
    num_workers=4
)

