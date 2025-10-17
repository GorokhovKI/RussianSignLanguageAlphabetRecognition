import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

ANNOTATIONS_FILE = Path("data/annotations.tsv")
OUTPUT_DIR = Path("results/dataset_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Loading {ANNOTATIONS_FILE}...")
try:
    df = pd.read_csv(ANNOTATIONS_FILE, sep='\t')
    print(f"[INFO] Loaded {len(df)} lines.")
except FileNotFoundError:
    print(f"[ERROR] File not found: {ANNOTATIONS_FILE}")
    exit()
except Exception as e:
    print(f"[ERROR] Error loading file: {e}")
    exit()

print("\n[INFO] Samples by class")
class_counts = df['text'].value_counts()
print(f"Len unique classes: {len(class_counts)}")
print("Top 10 classes by num of videos:")
print(class_counts.head(10))

print("Top 10 classes by num of videos asc:")
print(class_counts.tail(10))

#Histogramm
plt.figure(figsize=(14, 8))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Samples by class')
plt.xlabel('Class')
plt.ylabel('Videos num')
plt.xticks(rotation=90)
plt.tight_layout()
histogram_path = OUTPUT_DIR / "class_distribution.png"
plt.savefig(histogram_path)
print(f"[INFO] Sample by classes saved: {histogram_path}")
plt.close()

print("\n[INFO] Samples by resolution")
resolution_counts = df.groupby(['width', 'height']).size().reset_index(name='counts')
resolution_counts['resolution_str'] = resolution_counts['width'].astype(str) + 'x' + resolution_counts['height'].astype(str)

plt.figure(figsize=(10, 6))
sns.barplot(data=resolution_counts, x='resolution_str', y='counts')
plt.title('Samples by resolution')
plt.xlabel('Resolution')
plt.ylabel('Videos num')
plt.tight_layout()
plt.xticks(rotation=45)
plt.tight_layout()
res_hist_path = OUTPUT_DIR / "resolution_distribution.png"
plt.savefig(res_hist_path)
print(f"[INFO] Samples by video saved: {res_hist_path}")
plt.close()

print("\n[INFO] Video length")
lengths = df['length']
print(f"Statistic:")
print(f"Min: {lengths.min()}")
print(f"Max: {lengths.max()}")
print(f"Mean: {lengths.mean():.2f}")
print(f"Median: {lengths.median():.2f}")

plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, edgecolor='k', alpha=0.7)
plt.title('Videos by length')
plt.xlabel('Frames')
plt.ylabel('Videos num')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
length_hist_path = OUTPUT_DIR / "video_length_distribution.png"
plt.savefig(length_hist_path)
print(f"[INFO] Length video saved: {length_hist_path}")
plt.close()

print("\n[INFO] Train test")
train_test_counts = df['train'].value_counts()
print(f"Train: {train_test_counts.get(True, 0)} lines")
print(f"Test: {train_test_counts.get(False, 0)} lines")

plt.figure(figsize=(8, 5))
plt.pie(train_test_counts.values, labels=['Test', 'Train'], autopct='%1.1f%%', startangle=90)
plt.title('Train and Test')
plt.tight_layout()
train_test_pie_path = OUTPUT_DIR / "train_test_split.png"
plt.savefig(train_test_pie_path)
print(f"[INFO] Train/Test split saved: {train_test_pie_path}")
plt.close()

print("\n[INFO] Analysis saved")