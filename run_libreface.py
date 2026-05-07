import libreface

video_path = r"//gallery/video_me/newtry.mp4.mp4"

output_csv_path = r"//outputs/my_video_result.csv"

libreface.get_facial_attributes(
    video_path,
    output_save_path=output_csv_path,
    device="cpu",                 
    batch_size=18,                
    num_workers=4                 
)

print("Facial recognition done! Results saved to:", output_csv_path)
