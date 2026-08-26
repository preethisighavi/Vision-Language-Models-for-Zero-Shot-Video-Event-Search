import pandas as pd
import os
import shutil

df = pd.read_parquet('./data_local/outputs/intermediate/raw_unified.parquet')
vids = df[['video_id', 'source']].drop_duplicates().to_dict('records')

msrvtt_src = '../MSRVTT/raw_videos'
msvd_src = '../MSVD/raw_videos'

msrvtt_dst = './data_local/MSRVTT/raw_videos'
msvd_dst = './data_local/MSVD/raw_videos'

os.makedirs(msrvtt_dst, exist_ok=True)
os.makedirs(msvd_dst, exist_ok=True)

for row in vids:
    vid = row['video_id']
    src_dir = msrvtt_src if row['source'] == 'MSR-VTT' else msvd_src
    dst_dir = msrvtt_dst if row['source'] == 'MSR-VTT' else msvd_dst
    
    found = False
    for ext in ['.mp4', '.avi', '.mkv']:
        src_path = os.path.join(src_dir, f"{vid}{ext}")
        if os.path.exists(src_path):
            dst_path = os.path.join(dst_dir, f"{vid}{ext}")
            print(f"Copying {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)
            found = True
            break
    if not found:
        print(f"NOT FOUND: {vid}")
print("Done copying.")
