import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

#Path to data
DATASET_PATH = Path("data/original")
TRIMMED_DATASET_PATH = Path("data/trimmed")
ANNOTATIONS_PATH = Path("data/annotations.tsv")

#Path to save
OUTPUT_PATH = Path("results/dataset_analysis")
FEATURES_PATH = OUTPUT_PATH / "LANDMARKS"
EXAMPLE_IMAGES_PATH = OUTPUT_PATH / "example_image.JPG"

#Create dir's
FEATURES_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

#MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


#Initiazlization hands with default parameters
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    try:
        import pandas as pd
        #Reading annotation file(id of video + class (char)
        annotations_df = pd.read_csv(ANNOTATIONS_PATH, sep="\t")
        videos_ids = annotations_df["attachment_id"].tolist()
        videos_classes = annotations_df["text"].tolist()
        print("[INFO] Found {len(videos_id)} videos")
    except ImportError:
        print("[WARNING] Couldn't import pandas]")
        videos_ids = []
        videos_classes = []
        with open(ANNOTATIONS_PATH, "r") as annotations_file:
            lines = annotations_file.readlines()[1:] #Skip label
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    videos_ids.append(parts[0])
                    videos_classes.append(parts[2])
        print("[INFO] Found {len(videos_id)} videos by standard way")

    example_saved = False
    example_video_id = None
    example_frame_num = 0

    for idx, (video_id, class_label) in enumerate(zip(videos_ids, videos_classes)):
        video_path = TRIMMED_DATASET_PATH / f"{video_id}.mp4"

        if not video_path.exists():
            print(f"[INFO] Skipping {video_path}")
            continue

        print(f"[INFO] Обработка видео {idx + 1}/{len(videos_ids)}: {video_id} (Класс: {class_label})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Couldn't open video {video_path}")
            continue

        landmarks_per_video = []
        frame_num = 0

        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_landmarks = []

            #21 landmarks * 2 hands * 3 axis -1 hand as no hands
            flat_landmarks = [-1.0] * (21 * 3 * 2)

            if results.multi_hand_landmarks:
                hand_idx = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                        px, py, pz = landmark.x, landmark.y, landmark.z
                        # hand 1 + hand 2
                        start_idx = hand_idx * 21 * 3
                        flat_landmarks[start_idx + lm_idx * 3] = px
                        flat_landmarks[start_idx + lm_idx * 3 + 1] = py
                        flat_landmarks[start_idx + lm_idx * 3 + 2] = pz

                    hand_idx += 1
                    if hand_idx >= 2:  # Once more checking 2 hands
                        break

                # landmarks for current frame
            landmarks_per_video.append(flat_landmarks)

            # Saving example of mediapipe landmarks
            if not example_saved and results.multi_hand_landmarks and idx < len(
                    videos_ids) // 2:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                cv2.imwrite(str(EXAMPLE_IMAGES_PATH), frame)
                print(f"[INFO] Пример кадра с landmarks сохранен: {EXAMPLE_IMAGES_PATH}")
                example_saved = True
                example_video_id = video_id
                example_frame_num = frame_num

            frame_num += 1

        cap.release()
        if landmarks_per_video:
            output_filename = FEATURES_PATH / f"{video_id}_landmarks.npy"
            np.save(output_filename, np.array(landmarks_per_video, dtype=np.float32))
            print(f"Сохранено: {output_filename} (форма: {np.array(landmarks_per_video).shape})")
        else:
            print(f"Предупреждение: не удалось извлечь landmarks для {video_id}")

    if not example_saved:
        print("[WARNING] Не удалось найти кадр с обнаруженными руками для сохранения примера.")

print("\n[INFO] Извлечение признаков завершено.")







