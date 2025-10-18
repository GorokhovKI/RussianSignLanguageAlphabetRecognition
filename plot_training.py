import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

HISTORY_PATH = Path("results/model_training/training_history.pkl")
OUT_DIR = Path("results/model_training/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not HISTORY_PATH.exists():
    raise FileNotFoundError(f"Не найден файл истории обучения: {HISTORY_PATH}\n"
                            "Запустите train_landmarks.py прежде чем строить графики.")

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

# Валидация структуры
required_keys = {'train_loss','val_loss','train_acc','val_acc','train_f1','val_f1'}
if not isinstance(history, dict) or not required_keys.issubset(history.keys()):
    raise ValueError("Файл не содержит ожидаемой структуры history. Ожидаются ключи: " + ", ".join(required_keys))

epochs = list(range(1, len(history['train_loss']) + 1))
df = pd.DataFrame({
    'epoch': epochs,
    'train_loss': history['train_loss'],
    'val_loss': history['val_loss'],
    'train_acc': history['train_acc'],
    'val_acc': history['val_acc'],
    'train_f1': history['train_f1'],
    'val_f1': history['val_f1'],
})

# helper
def save_plot(x, y1, y2, title, ylabel, filename, legend):
    plt.figure(figsize=(9,5))
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(legend)
    plt.tight_layout()
    plt.savefig(str(filename))
    plt.close()
    print("Saved:", filename)

# Loss
save_plot(df['epoch'], df['train_loss'], df['val_loss'],
          title="Train vs Val Loss", ylabel="Loss",
          filename=OUT_DIR / "loss.png", legend=["train_loss", "val_loss"])

# Accuracy
save_plot(df['epoch'], df['train_acc'], df['val_acc'],
          title="Train vs Val Accuracy", ylabel="Accuracy",
          filename=OUT_DIR / "accuracy.png", legend=["train_acc", "val_acc"])

# F1
save_plot(df['epoch'], df['train_f1'], df['val_f1'],
          title="Train vs Val F1 (weighted)", ylabel="F1 score",
          filename=OUT_DIR / "f1.png", legend=["train_f1", "val_f1"])

# Save CSV summary and best epoch
df.to_csv(OUT_DIR / "training_history.csv", index=False)
best_idx = int(np.nanargmax(history['val_f1']))
best_epoch = best_idx + 1
best_row = df.loc[best_idx, ['epoch','val_loss','val_acc','val_f1']]
print("Best epoch (by val_f1):", best_epoch)
print(best_row)
