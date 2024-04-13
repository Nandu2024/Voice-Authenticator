import os
import time
import numpy as np
import pyaudio
import tensorflow as tf
import python_speech_features as psf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout# Importing Input function from tensorflow.keras.layers

# Global variables
DATA_DIR = "pathtofolder"
MODEL_WEIGHTS_FILE = "voice_authentication_model_weights.weights.h5"

# Recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
COUNTDOWN_SECONDS = 5

# CNN model parameters
INPUT_SHAPE = (13, 99, 1)  # MFCC feature shape (number of cepstral coefficients, number of frames, channels)

# Function to create the CNN model
def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Function to record audio
def record_audio(duration):
    frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

# Function to extract MFCC features
def extract_mfcc_features(data, input_shape):
    data = np.frombuffer(data, dtype=np.int16) / (2 ** 15)
    mfcc_features = psf.mfcc(data, RATE, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, preemph=0.97)
    
    # Resize the MFCC features to match the input shape
    mfcc_features = np.resize(mfcc_features, input_shape[:2])  # Resize to the input shape's first two dimensions
    mfcc_features = mfcc_features[:,:, np.newaxis]  # Add a channel dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    return mfcc_features

# Function to train the voice authenticator
def train_voice_authenticator():
    X = []
    y = []

    for user_folder in os.listdir(DATA_DIR):
        user_folder_path = os.path.join(DATA_DIR, user_folder)
        
        if os.path.isdir(user_folder_path):
            reference_sample_path = os.path.join(user_folder_path, "reference.wav")

            if os.path.isfile(reference_sample_path):
                # Load reference sample
                reference_data = record_audio(RECORD_SECONDS)
                reference_features = extract_mfcc_features(reference_data, INPUT_SHAPE)
                X.append(reference_features)
                y.append(1)  # Label as 1 indicating a reference sample

    X_train = np.array(X)
    y_train = np.array(y)

    model = create_cnn_model(INPUT_SHAPE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model weights
    model.save_weights(MODEL_WEIGHTS_FILE)

    return model

# Function to authenticate a user
def authenticate_user():
    print("Login attempt.")
    user_name = input("Enter your name: ")
    stored_data_path = os.path.join(DATA_DIR, user_name, "reference.wav")

    if os.path.isfile(stored_data_path):
        print("Please speak your password. Recording will start soon.")
        time.sleep(1)

        for i in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"Recording starts in {i} second(s)...")
            time.sleep(1)

        recorded_data = record_audio(RECORD_SECONDS)

        with open(stored_data_path, 'rb') as f:
            stored_data = f.read()

        similarity_score = authenticate(recorded_data, stored_data)

        if similarity_score >= 0.5:
            print("Login successful!")
        else:
            print("Login failed. Password mismatch.")
    else:
        print("User not found. Please check your name and try again.")

# Function to authenticate a user based on recorded data
def authenticate(recorded_data, stored_data):
    model = create_cnn_model(INPUT_SHAPE)
    model.load_weights(MODEL_WEIGHTS_FILE)

    recorded_features = extract_mfcc_features(recorded_data, INPUT_SHAPE)
    similarity_score = model.predict(recorded_features)[0][0]
    return similarity_score

# Function to create a new account
def create_account():
    print("Welcome! Let's create your account.")
    user_name = input("Enter your name: ")
    user_folder_path = os.path.join(DATA_DIR, user_name)
    reference_sample_path = os.path.join(user_folder_path, "reference.wav")

    print("Please speak your password. Recording will start soon.")
    time.sleep(1)

    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Recording starts in {i} second(s)...")
        time.sleep(1)

    reference_data = record_audio(RECORD_SECONDS)

    os.makedirs(user_folder_path, exist_ok=True)
    with open(reference_sample_path, 'wb') as f:
        f.write(reference_data)

    print("Recording saved as reference sample.")
    print("Training voice authenticator...")
    train_voice_authenticator()
    print("Voice authenticator trained successfully.")

# Main function
def main():
    while True:
        print("\nUser Menu:")
        print("1. Create Account")
        print("2. Login")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == "1":
            create_account()
        elif choice == "2":
            authenticate_user()
        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
