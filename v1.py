import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def train_music_generator(input_file, output_file, duration=60):
    """
    Asl musiqadan o'rganib, yangi musiqa yaratadi
    
    :param input_file: kiruvchi audio fayl
    :param output_file: chiqadigan audio fayl
    :param duration: yangi musiqaning davomiyligi (sekundlarda)
    """
    
    # 1. Audio faylni yuklash va tahlil qilish
    print("Audio faylni yuklash...")
    try:
        y, sr = librosa.load(input_file, duration=180)  # 3 daqiqalik audio
    except Exception as e:
        print(f"Audio faylni yuklashda xato: {e}")
        return
    
    # 2. Musiqa xususiyatlarini ajratib olish
    print("Xususiyatlarni ajratib olish...")
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    except Exception as e:
        print(f"Xususiyatlarni ajratishda xato: {e}")
        return
    
    # 3. Xususiyatlarni birlashtirish
    features = np.vstack([mfcc, chroma, spectral_contrast, tonnetz])
    features = features.T
    
    # 4. Ma'lumotlarni normalizatsiya qilish
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features_normalized = (features - mean) / (std + 1e-8)
    
    # 5. LSTM modelini yaratish
    print("Modelni yaratish...")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(512, input_shape=(100, features_normalized.shape[1])),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(features_normalized.shape[1], activation='tanh')
        ])
        
        model.compile(optimizer='adam', loss='mse')
    except Exception as e:
        print(f"Model yaratishda xato: {e}")
        return
    
    # 6. Ma'lumotlarni tayyorlash
    X, y = [], []
    seq_length = 100
    
    try:
        for i in range(len(features_normalized) - seq_length):
            X.append(features_normalized[i:i+seq_length])
            y.append(features_normalized[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
    except Exception as e:
        print(f"Ma'lumotlarni tayyorlashda xato: {e}")
        return
    
    # 7. Modelni o'qitish
    print("Modelni o'qitish...")
    try:
        model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    except Exception as e:
        print(f"Modelni o'qitishda xato: {e}")
        return
    
    # 8. Yangi musiqa yaratish
    print("Yangi musiqa yaratish...")
    try:
        generated = features_normalized[:seq_length]
        
        for _ in range(int(duration * sr / 512)):  # 512 - odatiy hop_length
            x = generated[-seq_length:].reshape(1, seq_length, features_normalized.shape[1])
            next_feat = model.predict(x, verbose=0)[0]
            generated = np.vstack([generated, next_feat])
    except Exception as e:
        print(f"Musiqa yaratishda xato: {e}")
        return
    
    # 9. Normalizatsiyani bekor qilish
    generated = generated * (std + 1e-8) + mean
    
    # 10. MFCC dan audio signalga o'tish
    print("Audio signalga aylantirish...")
    try:
        mfcc_gen = generated[:, :20].T
        audio = librosa.feature.inverse.mfcc_to_audio(mfcc_gen, sr=sr, n_iter=50)
    except Exception as e:
        print(f"Audio signalga aylantirishda xato: {e}")
        return
    
    # 11. Yangi audio faylni saqlash
    try:
        sf.write(output_file, audio, sr)
        print(f"Yangi musiqa saqlandi: {output_file}")
    except Exception as e:
        print(f"Faylni saqlashda xato: {e}")

if __name__ == "__main__":
    input_file = "input.mp3"  # O'zgartiring kerakli fayl nomiga
    output_file = "generated_music.wav"
    train_music_generator(input_file, output_file, duration=30)