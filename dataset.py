import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from tqdm import tqdm

def load_data(data_dir, img_size=(64, 64)):
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    images = []
    labels = []
    
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        print(f"Loading {cls_name}...", flush=True)
        for img_name in tqdm(os.listdir(cls_dir), desc=cls_name):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                if img.size != img_size:
                    img = img.resize(img_size)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                images.append(img_arr.flatten())
                labels.append(class_to_idx[cls_name])
                
    X = np.array(images)
    y = np.array(labels)
    
    return X, y, classes

def get_dataloaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15, random_state=42):
    X, y, classes = load_data(data_dir)
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )
    
    # Split train+val into train and val
    val_ratio_adjusted = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_train_val
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'classes': classes
    }

def batch_generator(X, y, batch_size, shuffle=True):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]
