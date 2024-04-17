import os
import time
import numpy as np
import pyaudio
import tensorflow as tf
import python_speech_features as psf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout# Importing Input function from tensorflow.keras.layers
from sklearn.metrics.pairwise import cosine_similarity
# Global variables
DATA_DIR = "path to folder"
MODEL_WEIGHTS_FILE = "voice_authentication_model_weights.weights.h5"


# Recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
COUNTDOWN_SECONDS = 5

SIMILARITY_THRESHOLD = 0.5

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

    X_train = X_train.reshape(-1, 13, 99, 1)

    model = create_cnn_model(INPUT_SHAPE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model weights
    model.save_weights(MODEL_WEIGHTS_FILE)

    return model

def calculate_similarity(recorded_features, stored_features):
    # Reshape features if needed (to match expected input shape)
    recorded_features = recorded_features.reshape(1, -1)
    stored_features = stored_features.reshape(1, -1)
    
    # Compute cosine similarity between the features
    similarity_score = cosine_similarity(recorded_features, stored_features)[0][0]
    return similarity_score
# Function to authenticate a user
# Function to authenticate a user
def authenticate_user():
    print("Login attempt. Please speak your name.")
    
    # Record the user's voice input
    recorded_data = record_audio(RECORD_SECONDS)
    
    # Extract MFCC features from the recorded voice input
    recorded_features = extract_mfcc_features(recorded_data, INPUT_SHAPE)
    
    # Initialize variables to store the most similar user and their similarity score
    most_similar_user = None
    highest_similarity_score = -1
    
    # Iterate through each user's stored reference sample to find the most similar user
    for user_folder in os.listdir(DATA_DIR):
        user_folder_path = os.path.join(DATA_DIR, user_folder)
        
        if os.path.isdir(user_folder_path):
            reference_sample_path = os.path.join(user_folder_path, "reference.wav")
            
            if os.path.isfile(reference_sample_path):
                # Load the stored reference sample of the current user
                with open(reference_sample_path, 'rb') as f:
                    stored_data = f.read()
                
                # Extract MFCC features from the stored reference sample
                stored_features = extract_mfcc_features(stored_data, INPUT_SHAPE)
                
                # Calculate similarity score between recorded and stored features
                similarity_score = calculate_similarity(recorded_features, stored_features)
                
                # Update most similar user and similarity score if needed
                if similarity_score > highest_similarity_score:
                    highest_similarity_score = similarity_score
                    most_similar_user = user_folder
    
    # Check if a user with sufficient similarity was found
    if highest_similarity_score >= SIMILARITY_THRESHOLD:
        print(f"Welcome back, {most_similar_user}!")
    else:
        print("User not recognized. Please try again.")

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
