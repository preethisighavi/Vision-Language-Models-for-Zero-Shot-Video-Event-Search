import cv2
import numpy as np
from PIL import Image

class VideoProcessor:
    def __init__(self, target_resolution=(224, 224), fps_downsample_rate=1):
        """
        :param target_resolution: Tuple (W, H) expected by ViT blocks (usually 224x224).
        :param fps_downsample_rate: Only yield 1 frame per exact [X] seconds of video.
        """
        self.target_res = target_resolution
        self.fps_downsample_rate = fps_downsample_rate
        
    def _center_crop_and_resize(self, frame):
        """Crop frames dynamically to maintain aspect ratio without stretching."""
        h, w = frame.shape[:2]
        
        # Calculate exactly how to center crop a perfect square
        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        
        cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        resized = cv2.resize(cropped, self.target_res)
        
        # Convert BGR (OpenCV native) to RGB (CLIP/PIL native)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def extract_frames(self, video_path):
        """
        Safely open an MP4 and extract downsampled RGB Image arrays.
        Will yield exactly `Duration_Seconds / fps_downsample_rate` frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30 # Fallback heuristic
            
        frame_interval = fps * self.fps_downsample_rate
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                processed_frame = self._center_crop_and_resize(frame)
                # Append as PIL Image because Huggingface processors expect PIL or specific numpy permutations
                frames.append(Image.fromarray(processed_frame))
                
            frame_idx += 1
            
        cap.release()
        return frames
