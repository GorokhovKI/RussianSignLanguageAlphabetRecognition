import os
import random
from pathlib import Path
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sympy.codegen.fnodes import dimension
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, f1_score

#Parametrs
FEATURES_DIR = Path("results/dataset_analysis/LANDMARKS_mc3")
ANNOTATIONS_FILE = Path("data/annotations.tsv")
OUTPUT_DIR = Path("results/model_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DROPOUT_RATE = 0.5
MAX_SEQ_LENGTH = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if torch.cuda.is_available() else False

SCALER_PATH = OUTPUT_DIR / "scaler.pkl"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.pkl"
SEED = 4312

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

print(f"[INFO] Loading annotations{ANNOTATIONS_FILE}")
df = pd.read_csv(ANNOTATIONS_FILE, sep="\t")
print(f"[INFO] All lines: {len(df)}")

unique_classes = sorted(df['text'].unique())
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
num_classes = len(unique_classes)
print(f"[INFO] Classes: {num_classes}")

# ------------------------
# == Dataset
# ------------------------
def replace_missing_and_add_presence_channel(arr: np.ndarray) -> np.ndarray:
    # arr: (num_frames, 126) with -1 for missing hands
    # create presence flag per frame: 1 if any coordinate != -1 else 0
    presence = (~np.all(arr == -1.0, axis=1)).astype(np.float32)[:, None]  # (num_frames,1)
    arr_fixed = arr.copy()
    arr_fixed[arr_fixed == -1.0] = 0.0
    return np.concatenate([arr_fixed, presence], axis=1)  # (num_frames, 127)

class LandmarksDataset(Dataset):
    def __init__(self, features_dir: Path, annotations_df: pd.DataFrame,
                 class_mapping: dict, max_length: int, scaler: dict = None, mode: str = "train"):
        """
        scaler: None or dict {'mean': np.array, 'std': np.array} shapes (feat,)
        mode: 'train'|'val'|'test' used only for logging
        """
        self.features_dir = Path(features_dir)
        self.annotations_df = annotations_df
        self.class_mapping = class_mapping
        self.max_length = max_length
        self.scaler = scaler
        self.mode = mode

        self.items = []
        for _, row in self.annotations_df.iterrows():
            vid = row['attachment_id']
            label = row['text']
            feature_file = self.features_dir / f"{vid}_landmarks.npy"
            if feature_file.exists():
                self.items.append((vid, label))
            else:
                # предупреждение, но не падаем
                print(f"[WARNING] {mode}: файл не найден: {feature_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        vid, label_str = self.items[index]
        label = self.class_mapping[label_str]
        fpath = self.features_dir / f"{vid}_landmarks.npy"
        arr = np.load(fpath)  # (num_frames, 126)

        arr = replace_missing_and_add_presence_channel(arr)  # (num_frames, 127)

        orig_len = arr.shape[0]
        #padding
        if orig_len > self.max_length:
            arr = arr[:self.max_length]
            seq_len = self.max_length
        else:
            pad = self.max_length - orig_len
            if pad > 0:
                arr = np.pad(arr, ((0, pad), (0, 0)), mode='constant', constant_values=0.0)
            seq_len = orig_len

        # normalize
        if self.scaler is not None:
            mean = self.scaler['mean']  # shape (feat,)
            std = self.scaler['std']
            arr = (arr - mean[None, :]) / (std[None, :] + 1e-9)

        # return tensors: (seq_len_padded, feat), label, orig_len
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(seq_len, dtype=torch.long)

# ------------------------
# == Collate fn
# ------------------------
def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    seqs = torch.stack(seqs)  # (batch, max_len, feat)
    labels = torch.stack(labels).long()
    lengths = torch.stack(lengths).long()

    # sort by length asc
    lengths_sorted, perm_idx = lengths.sort(descending=True)
    seqs = seqs[perm_idx]
    labels = labels[perm_idx]
    return seqs, labels, lengths_sorted


# ------------------------
# == Scaler computation (mean/std)
# ------------------------
def compute_mean_std_for_files(features_dir: Path, video_id_list: List[str]) -> dict:
    #summs, sumsq и count by frames
    feat_dim = 127
    s1 = np.zeros(feat_dim, dtype=np.float64)
    s2 = np.zeros(feat_dim, dtype=np.float64)
    n = 0
    for vid in tqdm(video_id_list, desc="Computing mean/std"):
        arr = np.load(Path(features_dir) / f"{vid}_landmarks.npy")  # (frames, 126)
        arr = replace_missing_and_add_presence_channel(arr)  # -> (frames,127)
        s1 += arr.sum(axis=0)
        s2 += (arr ** 2).sum(axis=0)
        n += arr.shape[0]
    mean = s1 / n
    var = (s2 / n) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}

# ------------------------
# CNN (time-preserving) + LSTM
# ------------------------
class AttentionPool(nn.Module):
    """Simple attention pooling over temporal axis.
       Input: (batch, seq_len, feat)
       Output: (batch, feat)
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.proj = nn.Linear(feat_dim, feat_dim // 2)
        self.v = nn.Linear(feat_dim // 2, 1)

    def forward(self, x, lengths):
        # x: (batch, T, feat)
        att = torch.tanh(self.proj(x))            # (batch, T, d)
        scores = self.v(att).squeeze(-1)         # (batch, T)
        # mask padded positions
        device = x.device
        max_len = x.size(1)
        mask = (torch.arange(max_len, device=device)[None, :] < lengths[:, None]).float()  # 1 for valid
        scores = scores.masked_fill(mask == 0, float('-1e9'))
        weights = torch.softmax(scores, dim=1)   # (batch, T)
        out = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, feat)
        return out

class LandmarksCNNLSTM(nn.Module):
    def __init__(self, input_feat=127, num_classes=32,
                 cnn_channels=[64, 128], hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()
        # temporal reduction factor = 2 * 2 = 4
        self.pool_factor = 4

        # conv block with BatchNorm + residual-ish small stack
        self.conv1 = nn.Conv1d(in_channels=input_feat, out_channels=cnn_channels[0],
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1],
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM -> more context
        self.lstm = nn.LSTM(input_size=cnn_channels[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)

        # Attention pooling (will pool outputs of LSTM)
        self.att_pool = AttentionPool(feat_dim=hidden_size * 2)

        # classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, lengths):
        # x: (batch, seq_len, feat)
        x = x.transpose(1, 2)             # -> (batch, feat, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)             # -> (batch, seq_len_reduced, channels)

        # compute reduced lengths (after pooling)
        lengths_reduced = ((lengths + (self.pool_factor - 1)) // self.pool_factor).long()
        # clamp to valid [1, x.size(1)]
        max_len = x.size(1)
        lengths_reduced = torch.clamp(lengths_reduced, min=1, max=max_len)

        # pack for LSTM
        packed = pack_padded_sequence(x, lengths_reduced.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)  # (batch, T_reduced, hidden*2)

        # attention pooling over time (using lengths_reduced)
        pooled = self.att_pool(out_unpacked, lengths_reduced.to(out_unpacked.device))  # (batch, hidden*2)

        logits = self.fc(pooled)
        return logits

# ------------------------
# == Validation
# ------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seqs, labels, lengths in dataloader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * seqs.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    if len(dataloader.dataset) == 0:
        return float('nan'), 0.0, 0.0
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_labels) else 0.0
    return avg_loss, acc, f1

# ------------------------
# == Main function
# ------------------------
def main():
    train_df = df[df['train'] == True].reset_index(drop=True)
    test_df = df[df['train'] == False].reset_index(drop=True)
    unique_users = train_df['user_id'].unique()
    val_user_count = max(1, int(0.1 * len(unique_users)))
    rng = np.random.default_rng(SEED)
    val_users = rng.choice(unique_users, size=val_user_count, replace=False)
    val_df = train_df[train_df['user_id'].isin(val_users)].reset_index(drop=True)
    train_df_final = train_df[~train_df['user_id'].isin(val_users)].reset_index(drop=True)

    print(f"[INFO] train rows: {len(train_df_final)}, val rows: {len(val_df)}, test rows: {len(test_df)}")

    # --- compute or load scaler from train videos ---
    if SCALER_PATH.exists():
        print(f"[INFO] Loading scaler from {SCALER_PATH}")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("[INFO] Calculating mean/std on train set")
        train_vids = train_df_final['attachment_id'].tolist()
        scaler = compute_mean_std_for_files(FEATURES_DIR, train_vids)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[INFO] Scaler saved: {SCALER_PATH}")

    # --- datasets & loaders ---
    train_dataset = LandmarksDataset(FEATURES_DIR, train_df_final, class_to_idx, max_length=MAX_SEQ_LENGTH, scaler=scaler, mode='train')
    val_dataset = LandmarksDataset(FEATURES_DIR, val_df, class_to_idx, max_length=MAX_SEQ_LENGTH, scaler=scaler, mode='val')
    test_dataset = LandmarksDataset(FEATURES_DIR, test_df, class_to_idx, max_length=MAX_SEQ_LENGTH, scaler=scaler, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # --- model, opt, criterion, amp scaler ---
    model = LandmarksCNNLSTM(input_feat=127, num_classes=num_classes, hidden_size=256, num_layers=2, dropout=DROPOUT_RATE)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # --- training loop ---
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for seqs, labels, lengths in pbar:
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(seqs, lengths)
                loss = criterion(logits, labels)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) else 0.0
        epoch_acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_labels) else 0.0

        # validation
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

        print(f"[E{epoch+1}] train loss={epoch_loss:.4f} acc={epoch_acc:.4f} f1={epoch_f1:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_f1'].append(epoch_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[INFO] Saved best model (val_f1={best_val_f1:.4f}) -> {BEST_MODEL_PATH}")

    print("[INFO] Training has been finished. Loading best model and testing.")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[RESULT] test loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)
    print(f"[INFO] History saved: {HISTORY_PATH}")

if __name__ == "__main__":
    main()
