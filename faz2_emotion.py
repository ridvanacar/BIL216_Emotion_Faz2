import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def extract_features(file_path):

    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    features = []

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features.append(zcr)

    rms = np.mean(librosa.feature.rms(y=y))
    features.append(rms)

    spec_cent = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )
    features.append(spec_cent)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    for value in mfcc_mean:
        features.append(value)

    return features


dataset_path = "Dataset"

X = []
y = []

emotions = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

for emotion in emotions:

    emotion_folder = os.path.join(dataset_path, emotion)

    for file_name in os.listdir(emotion_folder):

        if file_name.endswith(".wav"):

            file_path = os.path.join(
                emotion_folder,
                file_name
            )

            try:
                features = extract_features(file_path)

                X.append(features)
                y.append(emotion)

                print(f"Processed: {file_name}")

            except:
                print(f"ERROR: {file_name}")


columns = [
    "ZCR",
    "RMS",
    "Spectral_Centroid"
]

for i in range(13):
    columns.append(f"MFCC_{i+1}")

df = pd.DataFrame(X, columns=columns)

df["Emotion"] = y

df.to_csv("faz2_features.csv", index=False)

print("\nFeature CSV oluşturuldu.")

X = df.drop(columns=["Emotion"])
y = df["Emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n======================")
print(f"FAZ 2 ACCURACY: %{accuracy*100:.2f}")
print("======================\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")

plt.show()
