from pathlib import Path

from transformers import DistilBertTokenizer
from torchvision import transforms


class SpartanPreprocessor:
    def __init__(self, max_seq_length=32):
        self.max_seq_length = max_seq_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            Path(__file__).with_name("tokenizer"), local_files_only=True
        )
        self.vision_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def process_text(self, text):
        if isinstance(text, str):
            text = [text]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]
