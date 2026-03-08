import libreface

# Path to your input video (MP4, AVI, etc.)
video_path = r"//gallery/video_me/newtry.mp4.mp4"

# Path to save the CSV results (output per frame)
output_csv_path = r"//outputs/my_video_result.csv"

# Run facial analysis (uses ResNet-18 in libreface under the hood)
libreface.get_facial_attributes(
    video_path,
    output_save_path=output_csv_path,
    device="cpu",                 # Use "cpu" or "cuda:0" for GPU
    batch_size=18,                # Optional: tune for performance
    num_workers=4                 # Optional: tune for performance
)

print("Facial recognition done! Results saved to:", output_csv_path)
