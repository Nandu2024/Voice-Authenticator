import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.signal

# Global variables
DATA_DIR = "user_data"

# Recording parameters
CHUNK = 1024  # Samples to read at a time
FORMAT = pyaudio.paInt16  # Sample format (16 bits per sample)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (Hz)
RECORD_SECONDS = 5  # Recording duration
COUNTDOWN_SECONDS = 5  # Countdown duration for recording

def analyze_audio(data):
   
    # Convert data from signed 16-bit to float
    data = np.frombuffer(data, dtype=np.int16) / (2 ** 15)
    amplitude = data  # Assign data to amplitude

    # Calculate time axis
    time = np.arange(len(data)) / RATE

    # Find peaks and troughs by looking for sign changes
    peaks = np.diff(np.sign(data)) > 0
    troughs = np.diff(np.sign(data)) < 0

    # Identify maximum and minimum values
    max_amplitude = np.max(data)
    min_amplitude = np.min(data)

    return time, amplitude, peaks, troughs

def audio_similarity(recorded_data, stored_data, threshold=0.25):
    
    # Convert audio data to numpy arrays
    recorded_signal = np.frombuffer(recorded_data, dtype=np.int16).copy()
    stored_signal = np.frombuffer(stored_data, dtype=np.int16).copy()

    # Normalize signals
    recorded_signal_norm = recorded_signal.astype(np.float64) / np.max(np.abs(recorded_signal))
    stored_signal_norm = stored_signal.astype(np.float64) / np.max(np.abs(stored_signal))

    # Calculate cross-correlation
    correlation = scipy.signal.correlate(recorded_signal_norm, stored_signal_norm, mode='same')

    # Find the maximum correlation value
    max_correlation = np.max(correlation)

    # Compare with the threshold
    if max_correlation >= threshold:
        return True
    else:
        return False


def create_account():
    
    name = input("Enter your name: ")
    password_filename = os.path.join(DATA_DIR, f"{name}_password.wav")

    print("Please speak your password. Recording will start soon.")

    # Countdown for recording
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Recording starts in {i} second(s)...")
        time.sleep(1)

    frames = []

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Record audio data
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Combine audio frames
    data = b''.join(frames)

    # Save recorded audio as WAV file
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(password_filename, 'wb') as f:
        f.write(data)

    print(f"Voice password saved for user '{name}'.")

    # Analyze and plot the recorded audio
    t, amplitude, _, _ = analyze_audio(data)
    plt.figure()
    plt.plot(t, amplitude)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Recorded Voice Password")
    plt.grid(True)
    plt.show()

def login_attempt():
    
    print("Login attempt. Please speak your password. Recording will start soon.")

    # Countdown for recording
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Recording starts in {i} second(s)...")
        time.sleep(1)

    frames = []

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Record audio data
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Combine audio frames
    recorded_data = b''.join(frames)

    # Compare with stored passwords
    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_password.wav"):
            stored_password = os.path.join(DATA_DIR, filename)
            with open(stored_password, 'rb') as f:
                stored_data = f.read()

            # Check similarity using cross-correlation
            if audio_similarity(recorded_data, stored_data, threshold=0.25):
                print(f"Login successful! Welcome {filename.split('_')[0]}")
                return

    print("Login failed. Password mismatch or insufficient similarity.")

def user_menu():
    
    while True:
        print("User Menu:")
        print("1. Create Account")
        print("2. Login")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == "1":
            create_account()
        elif choice == "2":
            login_attempt()
        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    """
    Main function to demonstrate user menu functionality.
    """
    user_menu()

if __name__ == "__main__":
    main()
