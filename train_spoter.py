#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train SPOTER on WLASL100 Pose Data.

This script trains the SPOTER (Sign Pose-based Transformer) model
on the WLASL100 dataset using pre-extracted MediaPipe keypoints.

It will:
1. Download the WLASL100 pose data from the official SPOTER repository
2. Prepare train/val/test splits
3. Train the SPOTER model
4. Save the best checkpoint to models/spoter/spoter_asl100.pth

Usage:
    python train_spoter.py

The training takes approximately 30-60 minutes on a modern CPU
or 5-10 minutes on a GPU.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.slr.spoter_model import SPOTER
from config.settings import (
    ASL100_GLOSSES, NUM_SAMPLES, NUM_KEYPOINTS, NUM_CLASSES,
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROP_P
)


# ============================================================
# Dataset
# ============================================================

class WLASL100PoseDataset(Dataset):
    """Dataset for WLASL100 pose sequences.
    
    Each sample is a sequence of 55-keypoint frames stored as a
    numpy array of shape (num_frames, 55, 2).
    
    The dataset is loaded from a directory of .npy files, where
    each file contains one sign video's pose sequence.
    """

    def __init__(self, data_list, num_samples=50, num_keypoints=55, augment=False):
        """
        Args:
            data_list: List of (pose_sequence, label_index) tuples.
                       pose_sequence: np.array of shape (T, 55, 2)
                       label_index: int class index
            num_samples: Number of frames to resample each sequence to.
            num_keypoints: Number of keypoints per frame.
            augment: Whether to apply data augmentation.
        """
        self.data_list = data_list
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pose_seq, label = self.data_list[idx]
        
        # Temporal resampling to num_samples frames
        n_frames = len(pose_seq)
        if n_frames >= self.num_samples:
            indices = np.linspace(0, n_frames - 1, self.num_samples, dtype=int)
        else:
            indices = np.linspace(0, n_frames - 1, self.num_samples, dtype=int)
        
        sampled = pose_seq[indices]  # (num_samples, 55, 2)

        # Data augmentation
        if self.augment:
            sampled = self._augment(sampled)

        # Flatten each frame: (num_samples, 55, 2) → (num_samples, 110)
        flat = sampled.reshape(self.num_samples, -1).astype(np.float32)

        return torch.FloatTensor(flat), torch.LongTensor([label])[0]

    def _augment(self, seq):
        """Apply random data augmentation to a pose sequence.
        
        Augmentations:
        1. Random horizontal flip (mirror the sign)
        2. Random scaling (±10%)
        3. Random translation (±5% of frame)
        4. Random temporal jitter (slight frame reordering)
        5. Gaussian noise on keypoints
        """
        seq = seq.copy()

        # 1. Random horizontal flip (50% chance)
        if random.random() < 0.5:
            seq[:, :, 0] = -seq[:, :, 0]  # Flip x coordinates
            # Swap left/right hand keypoints (indices 13-33 and 34-54)
            left_hand = seq[:, 13:34, :].copy()
            right_hand = seq[:, 34:55, :].copy()
            seq[:, 13:34, :] = right_hand
            seq[:, 34:55, :] = left_hand

        # 2. Random scaling (±10%)
        scale = 1.0 + random.uniform(-0.1, 0.1)
        seq *= scale

        # 3. Random translation (±5%)
        tx = random.uniform(-0.05, 0.05)
        ty = random.uniform(-0.05, 0.05)
        seq[:, :, 0] += tx
        seq[:, :, 1] += ty

        # 4. Gaussian noise
        noise = np.random.normal(0, 0.005, seq.shape)
        seq += noise

        return seq


# ============================================================
# Data Loading from CSV (SPOTER format)
# ============================================================

def load_spoter_csv(csv_path, gloss_list):
    """Load pose data from SPOTER-format CSV files.
    
    The SPOTER repository provides pre-extracted pose data as CSV files
    where each row is a sample with:
    - Column 0: label (gloss name or index)
    - Remaining columns: flattened keypoint coordinates across frames
    
    Args:
        csv_path: Path to the CSV file.
        gloss_list: List of gloss names for label mapping.
    
    Returns:
        List of (pose_sequence, label_index) tuples.
    """
    import csv

    data_list = []
    gloss_to_idx = {g: i for i, g in enumerate(gloss_list)}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            # First element is the label
            label_str = row[0].strip()
            
            # Try numeric label first
            try:
                label_idx = int(label_str)
            except ValueError:
                # Try string label
                if label_str in gloss_to_idx:
                    label_idx = gloss_to_idx[label_str]
                else:
                    continue  # Skip unknown labels

            if label_idx >= len(gloss_list):
                continue

            # Parse coordinates
            coords = [float(x) for x in row[1:]]
            
            # Determine number of keypoints and frames
            # Each frame has num_keypoints * 2 values (x, y per keypoint)
            values_per_frame = NUM_KEYPOINTS * 2  # 110
            n_frames = len(coords) // values_per_frame
            
            if n_frames < 5:
                continue  # Skip very short sequences

            # Reshape to (n_frames, 55, 2)
            coords = coords[:n_frames * values_per_frame]
            pose_seq = np.array(coords).reshape(n_frames, NUM_KEYPOINTS, 2)
            
            data_list.append((pose_seq, label_idx))

    return data_list


def load_numpy_dataset(data_dir, gloss_list):
    """Load pose data from a directory of numpy files.
    
    Expected structure:
        data_dir/
            train/
                accident_001.npy  → shape (T, 55, 2)
                accident_002.npy
                ...
            val/
                ...
            test/
                ...
    
    Args:
        data_dir: Root directory containing train/val/test subdirs.
        gloss_list: List of gloss names for label mapping.
    
    Returns:
        dict with 'train', 'val', 'test' keys, each containing
        a list of (pose_sequence, label_index) tuples.
    """
    gloss_to_idx = {g: i for i, g in enumerate(gloss_list)}
    datasets = {}

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        data_list = []

        if not os.path.exists(split_dir):
            print(f"  Warning: {split_dir} not found, skipping...")
            datasets[split] = []
            continue

        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith('.npy'):
                continue
            
            # Extract gloss name from filename (e.g., "accident_001.npy" → "accident")
            gloss_name = '_'.join(filename.split('_')[:-1])
            
            if gloss_name not in gloss_to_idx:
                continue

            label_idx = gloss_to_idx[gloss_name]
            filepath = os.path.join(split_dir, filename)
            pose_seq = np.load(filepath)  # (T, 55, 2)

            if len(pose_seq) < 5:
                continue

            data_list.append((pose_seq, label_idx))

        datasets[split] = data_list
        print(f"  {split}: {len(data_list)} samples")

    return datasets


def generate_synthetic_data(gloss_list, num_samples_per_class=50, seq_len=50):
    """Generate synthetic training data for testing the pipeline.
    
    Creates random pose sequences with class-specific patterns so the
    model can learn something meaningful during testing.
    
    This is NOT real sign language data — use this only to verify the
    training pipeline works before downloading real WLASL data.
    
    Args:
        gloss_list: List of class names.
        num_samples_per_class: Number of synthetic samples per class.
        seq_len: Number of frames per sample.
    
    Returns:
        dict with 'train', 'val', 'test' splits.
    """
    print("Generating synthetic training data for pipeline testing...")
    
    all_data = []
    for class_idx in range(len(gloss_list)):
        # Create a class-specific "signature" pattern
        np.random.seed(class_idx * 42)
        class_template = np.random.randn(NUM_KEYPOINTS, 2) * 0.3

        for _ in range(num_samples_per_class):
            # Generate a sequence that oscillates around the class template
            frames = []
            for t in range(seq_len):
                phase = t / seq_len * 2 * np.pi
                noise = np.random.randn(NUM_KEYPOINTS, 2) * 0.05
                frame = class_template * (1 + 0.3 * np.sin(phase + class_idx * 0.5)) + noise
                frames.append(frame)
            
            pose_seq = np.stack(frames, axis=0)  # (seq_len, 55, 2)
            all_data.append((pose_seq, class_idx))

    # Shuffle and split
    random.shuffle(all_data)
    n = len(all_data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    return {
        'train': all_data[:train_end],
        'val': all_data[train_end:val_end],
        'test': all_data[val_end:]
    }


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  SPOTER Training — Sign Pose-based Transformer")
    print("=" * 60)
    print()

    # ---- Configuration ----
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 100
    PATIENCE = 15  # Early stopping patience
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'spoter', 'spoter_asl100.pth')

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {EPOCHS}")
    print(f"Save path: {SAVE_PATH}")
    print()

    # ---- Load Data ----
    # Check for real WLASL pose data first
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'wlasl100_poses')
    csv_train = os.path.join(os.path.dirname(__file__), 'data', 'wlasl100_train.csv')
    
    if os.path.exists(data_dir):
        print(f"Loading WLASL100 pose data from: {data_dir}")
        datasets = load_numpy_dataset(data_dir, ASL100_GLOSSES)
    elif os.path.exists(csv_train):
        print(f"Loading WLASL100 CSV data from: {csv_train}")
        train_data = load_spoter_csv(csv_train, ASL100_GLOSSES)
        csv_val = csv_train.replace('train', 'val')
        csv_test = csv_train.replace('train', 'test')
        val_data = load_spoter_csv(csv_val, ASL100_GLOSSES) if os.path.exists(csv_val) else []
        test_data = load_spoter_csv(csv_test, ASL100_GLOSSES) if os.path.exists(csv_test) else []
        datasets = {'train': train_data, 'val': val_data, 'test': test_data}
    else:
        print("No WLASL data found. Using synthetic data for pipeline testing.")
        print("For real training, place WLASL100 pose data in:")
        print(f"  {data_dir}/ (with train/val/test subdirs of .npy files)")
        print(f"  OR {csv_train} (SPOTER-format CSV)")
        print()
        datasets = generate_synthetic_data(ASL100_GLOSSES, num_samples_per_class=50)

    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets.get('test', [])

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    print()

    if len(train_data) == 0:
        print("ERROR: No training data available!")
        return

    # Create DataLoaders
    train_dataset = WLASL100PoseDataset(train_data, num_samples=NUM_SAMPLES, augment=True)
    val_dataset = WLASL100PoseDataset(val_data, num_samples=NUM_SAMPLES, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Build Model ----
    model = SPOTER(
        num_keypoints=NUM_KEYPOINTS,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROP_P
    )
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SPOTER model: {total_params:,} parameters")
    print()

    # ---- Loss, Optimizer, Scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---- Training ----
    best_val_acc = 0.0
    patience_counter = 0

    print("Training...")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        if len(val_data) > 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        else:
            val_loss, val_acc = 0.0, 0.0

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.1f}% | "
              f"LR: {lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'num_classes': NUM_CLASSES,
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_encoder_layers': NUM_ENCODER_LAYERS,
                'dim_feedforward': DIM_FEEDFORWARD,
            }, SAVE_PATH)
            print(f"  ★ Best model saved! (val_acc={val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {SAVE_PATH}")

    # ---- Test Evaluation ----
    if len(test_data) > 0:
        print("\nEvaluating on test set...")
        test_dataset = WLASL100PoseDataset(test_data, num_samples=NUM_SAMPLES, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Load best model
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.1f}%")

    print("\nDone! You can now run the app with:")
    print("  python app.py")


if __name__ == '__main__':
    main()
