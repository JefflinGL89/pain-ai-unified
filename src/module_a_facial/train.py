"""
Model Training Script for ConvNeXt-Tiny Facial Regressor
Trained on UNBC-McMaster Shoulder Pain Dataset with Subject-Independent Split
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FacialPainDataset, get_transforms
from model import ConvNeXtPainRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = ROOT_DIR / "dataset" / "pain_dataset_balanced.csv"
    checkpoint_dir = ROOT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Subject-independent Split: 80% train subjects, 20% val subjects
    unique_subjects = list(df['subject_id'].unique())
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    split_idx = int(len(unique_subjects) * 0.8)

    train_subs = unique_subjects[:split_idx]
    val_subs = unique_subjects[split_idx:]

    train_df = df[df['subject_id'].isin(train_subs)]
    val_df = df[df['subject_id'].isin(val_subs)]

    print(f"Train subjects ({len(train_subs)}): {len(train_df)} frames")
    print(f"Val subjects   ({len(val_subs)}): {len(val_df)} frames")

    train_dataset = FacialPainDataset(train_df, transform=get_transforms('train'))
    val_dataset = FacialPainDataset(val_df, transform=get_transforms('val'))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvNeXtPainRegressor(pretrained=True).to(device)

    # Huber Loss is less sensitive to noisy pain ground-truth annotations
    criterion = nn.HuberLoss(delta=0.1)

    # Differential learning rates: train regressor faster than backbone
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-2)

    epochs = 10
    best_val_mae = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)

                # Scale back up to original 0-16 PSPI scale
                val_preds.extend((preds * 16.0).cpu().numpy().flatten())
                val_targets.extend((labels * 16.0).numpy().flatten())

        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Train Huber Loss: {train_loss:.4f}")
        print(f"Val MAE (0-16 scale):  {val_mae:.4f}")
        print(f"Val RMSE (0-16 scale): {val_rmse:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = checkpoint_dir / "best_convnext_pain_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Best model saved to: {save_path}\n")


if __name__ == "__main__":
    train_model()
