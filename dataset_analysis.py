import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def analyze_aider_dataset(dataset_path="dataset/aider_ dataset/"):
    """Analyze the AIDER dataset structure and properties"""
    
    print("=== AIDER Dataset Analysis ===\n")
    
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found {len(class_dirs)} disaster classes:")
    
    class_counts = {}
    image_info = []
    
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(image_files)
        
        print(f"  {class_name}: {len(image_files)} images")
        
        for i, img_file in enumerate(image_files[:5]):
            try:
                img_path = os.path.join(class_path, img_file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_info.append({
                        'class': class_name,
                        'filename': img_file,
                        'width': width,
                        'height': height,
                        'aspect_ratio': width/height
                    })
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    df = pd.DataFrame(image_info)
    
    print(f"\nTotal images: {sum(class_counts.values())}")
    print(f"Average images per class: {np.mean(list(class_counts.values())):.1f}")
    
    print("\n=== Class Distribution ===")
    total_images = sum(class_counts.values())
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} images ({percentage:.1f}%)")
    
    if not df.empty:
        print("\n=== Image Size Analysis ===")
        print(f"Width range: {df['width'].min()} - {df['width'].max()}")
        print(f"Height range: {df['height'].min()} - {df['height'].max()}")
        print(f"Average aspect ratio: {df['aspect_ratio'].mean():.2f}")
        print(f"Most common sizes:")
        size_counts = Counter(zip(df['width'], df['height']))
        for (w, h), count in size_counts.most_common(5):
            print(f"  {w}x{h}: {count} images")
    
    print("\n=== Recommendations for UAV Deployment ===")
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 5:
        print("⚠️  High class imbalance detected (ratio: {:.1f}:1)".format(imbalance_ratio))
        print("   Consider using weighted loss or data augmentation")
    
    disaster_classes = [c for c in class_counts.keys() if c != 'normal']
    print(f"\nSuggested initial classes for UAV disaster detection:")
    print(f"  - Binary: normal vs disaster (all non-normal classes)")
    print(f"  - Multi-class: {', '.join(disaster_classes[:3])}")
    
    print(f"\nModel recommendations:")
    print(f"  - MobileNetV2: Good balance of accuracy and speed")
    print(f"  - EfficientNet-B0: Better accuracy, slightly slower")
    print(f"  - Target input size: 224x224 (standard for these models)")
    
    return class_counts, df

def create_class_mapping(dataset_path="dataset/aider_ dataset/", focus_classes=None):
    """Create class mapping for training"""
    
    all_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if focus_classes is None:
        focus_classes = ['fire', 'normal']
    
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(focus_classes))}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    print(f"\nClass mapping for training:")
    for cls, idx in class_to_idx.items():
        print(f"  {cls}: {idx}")
    
    return class_to_idx, idx_to_class

if __name__ == "__main__":
    class_counts, df = analyze_aider_dataset()
    
    print("\n" + "="*50)
    print("TRAINING SCENARIOS")
    print("="*50)
    
    print("\n1. Binary Fire Detection:")
    fire_mapping, _ = create_class_mapping(focus_classes=['fire', 'normal'])
    
    print("\n2. Multi-class Disaster Detection:")
    disaster_mapping, _ = create_class_mapping(focus_classes=['fire', 'collapsed_building', 'flooded_areas', 'normal'])
    
    print("\n3. All Classes:")
    all_mapping, _ = create_class_mapping(focus_classes=['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident', 'normal']) 