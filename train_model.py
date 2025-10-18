import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle

# --- Параметры ---
# Путь к результатам извлечения признаков
FEATURES_DIR = Path("results/dataset_analysis/LANDMARKS_mc3")
ANNOTATIONS_FILE = Path("data/annotations.tsv")
OUTPUT_DIR = Path("results/model_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Гиперпараметры
BATCH_SIZE = 64
LEARNING_RATE = 0.000001
NUM_EPOCHS = 100
DROPOUT_RATE = 0.5
MAX_SEQ_LENGTH = 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используется устройство: {DEVICE}")

# --- Загрузка аннотаций ---
print(f"[INFO] Загружаем аннотации из {ANNOTATIONS_FILE}...")
try:
    df = pd.read_csv(ANNOTATIONS_FILE, sep='\t')
    print(f"[INFO] Успешно загружено {len(df)} записей.")
except FileNotFoundError:
    print(f"[ERROR] Файл не найден: {ANNOTATIONS_FILE}")
    exit()

# --- Подготовка меток ---
unique_classes = sorted(df['text'].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
num_classes = len(unique_classes)
print(f"[INFO] Найдено {num_classes} уникальных классов.")


class LandmarksDataset(Dataset):
    def __init__(self, features_dir, annotations_df, class_mapping, max_length):
        self.features_dir = Path(features_dir)
        self.annotations_df = annotations_df
        self.class_mapping = class_mapping
        self.max_length = max_length

        # Фильтруем файлы, которые есть в аннотациях и существуют на диске
        self.video_ids = []
        for _, row in self.annotations_df.iterrows():
            video_id = row['attachment_id']
            label = row['text']
            feature_file = self.features_dir / f"{video_id}_landmarks.npy"
            if feature_file.exists():
                self.video_ids.append((video_id, label))
            else:
                print(f"[WARNING] Файл признаков не найден: {feature_file}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id, label_str = self.video_ids[idx]
        label = self.class_mapping[label_str]

        feature_file = self.features_dir / f"{video_id}_landmarks.npy"
        landmarks = np.load(feature_file)  # Shape: (num_frames, 126)

        # --- Выравнивание последовательности ---
        if len(landmarks) > self.max_length:
            # Усечение: берем первые max_length кадров
            # (или случайный сегмент, если хотите)
            landmarks = landmarks[:self.max_length]
        elif len(landmarks) < self.max_length:
            # Дополнение: нулями до max_length
            # (или повторением последнего кадра)
            pad_width = self.max_length - len(landmarks)
            padded_landmarks = np.pad(landmarks, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            landmarks = padded_landmarks
        # Теперь длина всегда self.max_length

        # Конвертируем в тензор
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)  # Shape: (MAX_SEQ_LENGTH, 126)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return landmarks_tensor, label_tensor  # Shape: (150, 126), ()


# --- Модель CNN + LSTM (обновленная) ---
class LandmarksCNNLSTM(nn.Module):
    def __init__(self, input_size=126, num_classes=32, hidden_size=256, num_layers=2, dropout_rate=0.5):
        super(LandmarksCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN слой для извлечения признаков из каждого кадра landmarks
        # Вход: (batch, seq_len, 126) -> Transpose -> (batch, 126, seq_len)
        # -> Conv1d -> (batch, 64, new_seq_len) -> ...
        # Это позволяет CNN "видеть" паттерны в 126 признаках каждого кадра
        # и как они меняются во времени (seq_len).
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=126, out_channels=64, kernel_size=3, padding=1),  # Вход: 126 признаков на кадр
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Уменьшаем временную размерность
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
            # Добавим еще один слой для большего сжатия
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=50)  # Адаптивно усредняем до 50 временных шагов
        )

        # После CNN, выход будет (batch, 256, 50)
        # LSTM ожидает (batch, seq_len, features), поэтому транспонируем
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate)

        # Полносвязный слой для классификации
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch, seq_len, 126) -> после CNN -> (batch, 256, 50) -> transpose -> (batch, 50, 256)
        x = x.transpose(1, 2)  # (batch, 126, seq_len)
        x = self.cnn(x)  # (batch, 256, 50)
        x = x.transpose(1, 2)  # (batch, 50, 256)

        # Применяем LSTM
        lstm_out, (hidden, _) = self.lstm(x)  # hidden: (num_layers, batch, hidden_size)

        # Используем последнее скрытое состояние LSTM для классификации
        last_hidden = self.dropout(hidden[-1])  # (batch, hidden_size)
        output = self.fc(last_hidden)  # (batch, num_classes)
        return output


# --- Функция валидации ---
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            preds = output.argmax(dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets)
    val_loss /= len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    return val_loss, val_acc, val_f1


# --- Основной цикл обучения ---
def main():
    # --- Подготовка данных ---
    # Загрузка данных с учетом разделения train/test
    train_df = df[df['train'] == True]
    test_df = df[df['train'] == False]

    # Для более строгого разделения по user_id (как в оригинальном датасете)
    # можно разделить уникальные user_id, а не строки датафрейма.
    # Пока используем стандартное разделение train/test из аннотаций.
    # Допустим, мы хотим выделить часть из train_df для валидации.
    unique_train_users = train_df['user_id'].unique()
    val_users = np.random.choice(unique_train_users, size=int(0.1 * len(unique_train_users)), replace=False)
    val_df = train_df[train_df['user_id'].isin(val_users)]
    train_df_final = train_df[~train_df['user_id'].isin(val_users)]

    print(f"[INFO] Размер обучающей выборки (после удаления val): {len(train_df_final)}")
    print(f"[INFO] Размер валидационной выборки (из train): {len(val_df)}")
    print(f"[INFO] Размер тестовой выборки (из аннотаций): {len(test_df)}")

    # Создание датасетов с выравниванием
    train_dataset = LandmarksDataset(FEATURES_DIR, train_df_final, class_to_idx, max_length=MAX_SEQ_LENGTH)
    val_dataset = LandmarksDataset(FEATURES_DIR, val_df, class_to_idx, max_length=MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Создание модели, оптимизатора, критерия ---
    model = LandmarksCNNLSTM(num_classes=num_classes, dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- Обучение ---
    train_losses = []
    train_accuracies = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_f1s = []

    best_val_f1 = 0.0
    best_model_path = OUTPUT_DIR / "best_model.pth"

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = output.argmax(dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(targets)

        # Метрики на эпохе
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        print(f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f}")

        # Валидация
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Сохранение истории
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        train_f1s.append(epoch_train_f1)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        # Сохранение лучшей модели по F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Сохранена лучшая модель с Val F1: {best_val_f1:.4f}")

    print("\n[INFO] Обучение завершено.")
    print(f"[INFO] Лучшая Val F1: {best_val_f1:.4f}, модель сохранена в {best_model_path}")

    # Сохранение истории обучения
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'train_f1': train_f1s,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
        'val_f1': val_f1s
    }
    history_path = OUTPUT_DIR / "training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"[INFO] История обучения сохранена: {history_path}")


if __name__ == "__main__":
    main()