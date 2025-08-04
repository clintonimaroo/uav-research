import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional

class AIDERDataset(Dataset):
    """PyTorch dataset for AIDER disaster detection dataset"""
    
    def __init__(self, 
                 dataset_path: str = "data/aider_ dataset/",
                 classes: List[str] = None,
                 transform: Optional[transforms.Compose] = None,
                 split: str = 'train',
                 train_ratio: float = 0.8):
        """
        Args:
            dataset_path: Path to the AIDER dataset
            classes: List of classes to include (default: ['fire', 'normal'])
            transform: Torchvision transforms to apply
            split: 'train', 'val', or 'all'
            train_ratio: Ratio of data to use for training
        """
        
        self.dataset_path = dataset_path
        self.classes = classes or ['fire', 'normal']
        self.transform = transform
        self.split = split
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        self.image_paths = []
        self.labels = []
        
        self._load_data()
        
        if split != 'all':
            self._split_data(train_ratio)
    
    def _load_data(self):
        """Load all image paths and labels"""
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} not found")
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.image_paths)} images across {len(self.classes)} classes")
        for cls in self.classes:
            count = self.labels.count(self.class_to_idx[cls])
            print(f"  {cls}: {count} images")
    
    def _split_data(self, train_ratio: float):
        """Split data into train/validation sets"""
        class_indices = {}
        for idx, label in enumerate(self.labels):
            class_name = self.idx_to_class[label]
            if class_name not in class_indices:
                class_indices[class_name] = []
            class_indices[class_name].append(idx)
        
        train_indices = []
        val_indices = []
        
        for class_name, indices in class_indices.items():
            np.random.shuffle(indices)
            n_train = int(len(indices) * train_ratio)
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])
        
        if self.split == 'train':
            selected_indices = train_indices
        else:  # 'val'
            selected_indices = val_indices
        
        self.image_paths = [self.image_paths[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        
        print(f"Split: {self.split} - {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalanced dataset"""
        class_counts = torch.zeros(len(self.classes))
        for label in self.labels:
            class_counts[label] += 1
        
        total_samples = len(self.labels)
        weights = total_samples / (len(self.classes) * class_counts)
        
        return weights

def get_transforms(input_size: int = 224, augment: bool = True):
    """Get image transforms for training and validation"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(dataset_path: str = "data/aider_ dataset/",
                       classes: List[str] = None,
                       batch_size: int = 32,
                       input_size: int = 224,
                       num_workers: int = 4,
                       augment: bool = True,
                       train_ratio: float = 0.8):
    """Create train and validation data loaders"""
    
    classes = classes or ['fire', 'normal']
    
    train_transform, val_transform = get_transforms(input_size, augment)
    
    train_dataset = AIDERDataset(
        dataset_path=dataset_path,
        classes=classes,
        transform=train_transform,
        split='train',
        train_ratio=train_ratio
    )
    
    val_dataset = AIDERDataset(
        dataset_path=dataset_path,
        classes=classes,
        transform=val_transform,
        split='val',
        train_ratio=train_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx

def test_dataset():
    """Test the dataset implementation"""
    print("Testing AIDER Dataset...")
    
    train_loader, val_loader, class_to_idx = create_data_loaders(
        classes=['fire', 'normal'],
        batch_size=4,
        num_workers=0  
    )
    
    print(f"Class mapping: {class_to_idx}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Label range: {labels.min().item()} - {labels.max().item()}")
        break
    
    print("Dataset test completed!")

if __name__ == "__main__":
    test_dataset() 