import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

data_dir = '/Users/spartan/Desktop/DATA298A'

def load_data():
    all_data = []
    for file in ['msrvtt_test_1k.json', 'msrvtt_train_7k.json', 'msrvtt_train_9k.json']:
        try:
            with open(os.path.join(data_dir, file), 'r') as f:
                all_data.extend(json.load(f))
        except FileNotFoundError:
             print(f"File not found {file}")
    return all_data

def analyze_dataset():
    data = load_data()
    df = pd.DataFrame(data)
    
    # Pre-configure beautiful seaborn styles
    sns.set_theme(style="whitegrid")
    
    # 1. Textual EDA: Caption Lengths & Top Words
    df['caption_length'] = df['caption'].apply(lambda x: len(word_tokenize(str(x))))
    
    stop_words = set(stopwords.words('english'))
    all_words = []
    
    for caption in df['caption']:
        words = word_tokenize(str(caption).lower())
        words = [w for w in words if w.isalnum() and w not in stop_words]
        all_words.extend(words)
        
    word_freq = Counter(all_words)
    top_words_df = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])
    
    # 2. Visual EDA: Sub-sampling Video files
    video_dir = os.path.join(data_dir, 'TrainValVideo')
    videos = list(set(df['video_id']))
    sample_videos = videos[:500] 
    
    video_metrics = []
    for vid in sample_videos:
        path = os.path.join(video_dir, f"{vid}.mp4")
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            video_metrics.append({
                'video_id': vid,
                'fps': fps,
                'aspect_ratio': f"{int(width)}x{int(height)}",
                'duration': frame_count / fps if fps > 0 else 0
            })
            
    vdf = pd.DataFrame(video_metrics)
    
    # ==========================
    # Plotting (Simplified / Meaningful Labels)
    # ==========================
    
    # Generate Category Mapping for readable charts
    category_mapping = {
        0: 'Music', 1: 'People', 2: 'Gaming', 3: 'Sports/Action', 4: 'News/Events', 
        5: 'Education', 6: 'TV Shows', 7: 'Movie Scenes', 8: 'Animation', 9: 'Vehicles/Transportation', 
        10: 'How-to/DIY', 11: 'Travel/Places', 12: 'Science/Tech', 13: 'Animals/Pets', 14: 'Kids/Family', 
        15: 'Documentary', 16: 'Food/Drink', 17: 'Cooking', 18: 'Beauty/Fashion', 19: 'Advertisement'
    }
    
    df['Category Name'] = df['category'].map(category_mapping)
    category_counts = df.drop_duplicates(subset=['video_id'])['Category Name'].value_counts()
    
    # 0. CATEGORY DISTRIBUTION
    plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.values, y=category_counts.index, palette='Blues_r')
    plt.title('Video Category Distribution', fontsize=16, pad=15)
    plt.xlabel('Number of Unique Videos', fontsize=12)
    plt.ylabel('Video Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'plot_category.png'), dpi=150)
    plt.close()

    # 1. DURATION
    plt.figure(figsize=(10, 5))
    sns.histplot(vdf['duration'], bins=20, color='#3498db', edgecolor='black')
    avg_dur = vdf['duration'].mean()
    plt.axvline(avg_dur, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_dur:.1f}s')
    plt.title('Video Length Distribution', fontsize=16, pad=15)
    plt.xlabel('Video Duration (seconds)', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'plot_duration.png'), dpi=150)
    plt.close()
    
    # 2. RESOLUTION
    plt.figure(figsize=(10, 5))
    top_res = vdf['aspect_ratio'].value_counts().head(5)
    sns.barplot(x=top_res.index, y=top_res.values, palette='viridis')
    plt.title('Top 5 Video Resolutions', fontsize=16, pad=15)
    plt.xlabel('Resolution (Width x Height)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'plot_resolution.png'), dpi=150)
    plt.close()
    
    # 3. FPS
    plt.figure(figsize=(10, 5))
    sns.histplot(vdf['fps'], bins=15, color='#e67e22', edgecolor='black')
    plt.title('Frame Rate (FPS) Distribution', fontsize=16, pad=15)
    plt.xlabel('Frames Per Second', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'plot_fps.png'), dpi=150)
    plt.close()
    
    # 4. SENTENCE LENGTH
    plt.figure(figsize=(10, 5))
    sns.histplot(df['caption_length'], bins=30, color='#2ecc71', edgecolor='black')
    plt.axvline(77, color='red', linestyle='dashed', linewidth=2, label='CLIP Max Word Limit (77)')
    plt.title('Caption Length Distribution (Words per query)', fontsize=16, pad=15)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'plot_sentence_length.png'), dpi=150)
    plt.close()
    
    # 5. TOP WORDS (Replacing WordCloud)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=top_words_df, palette='magma')
    plt.title('Top 15 Most Frequent Caption Words', fontsize=16, pad=15)
    plt.xlabel('Word Occurrence Count', fontsize=12)
    plt.ylabel('Vocabulary Word', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'plot_top_words.png'), dpi=150)
    plt.close()
    
    # Generate JSON Metrics
    metrics = {
        'total_videos': len(videos),
        'total_captions': len(df),
        'avg_words_per_caption': float(df['caption_length'].mean()),
        'pct_under_77_tokens': float((df['caption_length'] <= 77).mean() * 100),
        'most_common_resolution': str(vdf['aspect_ratio'].mode()[0]),
        'avg_fps': float(vdf['fps'].mean()),
        'avg_duration': float(vdf['duration'].mean())
    }
    
    with open(os.path.join(data_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    print("Starting Simplified EDA Extraction...")
    analyze_dataset()
    print("EDA completed successfully. Metrics and clearer plots saved to disk.")
