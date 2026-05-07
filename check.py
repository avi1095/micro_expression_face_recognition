import libreface
import os

video_path = r"/\gallery\video_me\newtry.avi"
output_csv_path = "//outputs/my_video_result_me.csv"

libreface.get_facial_attributes(
    video_path,
    output_save_path=output_csv_path,
    device="cuda:0",      
    batch_size=96,
    num_workers=6
)
print("CSV file exists?", os.path.exists(output_csv_path))
