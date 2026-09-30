import webdataset as wds
import torch
from torchvision import transforms
from transformers import AutoTokenizer
import numpy as np

def analyze_vision():
    print("========================================")
    print("--- Vision Data Statistics (MSR-VTT) ---")
    try:
        dataset = wds.WebDataset("data/msrvtt_shards/msrvtt-test-0000.tar").decode("pil")
        
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        raw_means = []
        raw_stds = []
        raw_shapes = set()
        
        prep_means = []
        prep_stds = []
        prep_shapes = set()
        
        count = 0
        for sample in dataset:
            if count >= 500: break
            raw_img = sample.get("f00.jpg")
            if not raw_img: continue
            
            # Raw stats (normalized to 0-1 for stat comparison)
            raw_np = np.array(raw_img) / 255.0 
            raw_means.append(raw_np.mean(axis=(0, 1)))
            raw_stds.append(raw_np.std(axis=(0, 1)))
            raw_shapes.add(raw_np.shape)
            
            # Preprocessed stats
            prep_tensor = transform(raw_img)
            prep_means.append(prep_tensor.mean(dim=(1, 2)).numpy())
            prep_stds.append(prep_tensor.std(dim=(1, 2)).numpy())
            prep_shapes.add(tuple(prep_tensor.shape))
            
            count += 1
            
        print(f"Analyzed {count} images directly from shard.")
        print("\n[ BEFORE PREPROCESSING ]")
        print(f"- Image Shapes: Highly Varied, e.g., {list(raw_shapes)[:3]}...")
        print(f"- Mean Pixel Intensity (RGB): {np.mean(raw_means, axis=0).round(3)}")
        print(f"- Std Dev Pixel Intensity (RGB): {np.mean(raw_stds, axis=0).round(3)}")
        
        print("\n[ AFTER PREPROCESSING ]")
        print(f"- Tensor Shapes: Strictly enforced {list(prep_shapes)}")
        print(f"- Mean Pixel Intensity (RGB): {np.mean(prep_means, axis=0).round(3)}")
        print(f"- Std Dev Pixel Intensity (RGB): {np.mean(prep_stds, axis=0).round(3)}")
        print("========================================")
        
    except Exception as e:
        print(f"Error Vision: {e}")

def analyze_text():
    print("\n========================================")
    print("--- Text Data Statistics (MSR-VTT) ---")
    try:
        dataset = wds.WebDataset("data/msrvtt_shards/msrvtt-test-0000.tar").decode()
        # Fix: Using openai/clip-vit-base-patch32 tokenizer instead of BERT
        # to ensure EDA accurately reflects the BPE tokenization limits that the CLIP model will enforce.
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        raw_lengths = []
        prep_lengths = []
        
        count = 0
        for sample in dataset:
            if count >= 1000: break
            txt = sample.get("txt")
            if not txt: continue
            
            raw_lengths.append(len(txt.split()))
            
            # Simulated Preprocessing
            tokens = tokenizer(txt, padding="max_length", truncation=True, max_length=77)
            prep_lengths.append(len(tokens["input_ids"]))
            
            count += 1
            
        print(f"Analyzed {count} text samples directly from shard.")
        print("\n[ BEFORE PREPROCESSING ]")
        print(f"- Average Word Count: {np.mean(raw_lengths):.2f}")
        print(f"- Max Word Count: {np.max(raw_lengths)}")
        print(f"- Memory Footprint: Variable per batch (inefficient)")
        
        print("\n[ AFTER PREPROCESSING (Tokenization & Truncation to 77) ]")
        print(f"- Tensor Sequence Length: {np.mean(prep_lengths):.2f}")
        print(f"- Tensor Shape: Strictly enforced [BatchSize, 77]")
        print("========================================")
        
    except Exception as e:
        print(f"Error Text: {e}")

if __name__ == "__main__":
    analyze_vision()
    analyze_text()
