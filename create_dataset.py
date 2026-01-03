import numpy as np
import librosa
import soundfile as sf
import os

def feature_extraction(input_audio_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.

    Parameters:
    input_audio_path (str): Path to the input audio file.
    n_mfcc (int): Number of MFCC features to extract.

    Returns:
    np.ndarray: Extracted MFCC features.
    """

    # Load the audio file
    audio_data, sr = librosa.load(input_audio_path, sr=None)

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio_data,
                                        sr=sr, 
                                        n_mfcc=n_mfcc)

    # Mean and standard deviation
    mfcc_mean = np.mean(mfcc_features, axis=1)
    mfcc_std = np.std(mfcc_features, axis=1)
    features = np.hstack((mfcc_mean, mfcc_std))

    return features

def build_dataset(feature_dir):
    """
    Extract features from all audio files in the given directory and create x and y datasets.
    Parameters:
    feature_dir (str): Path to the directory containing audio files.
    Returns:
    np.ndarray: Feature dataset (X).
    np.ndarray: Label dataset (y).
    """
    x_data = []
    y_data = []

    for class_name in os.listdir(feature_dir):
        folder_path = os.path.join(feature_dir, class_name)
        if os.path.isdir(folder_path):
            for audio_file in os.listdir(folder_path):
                audio_file_path = os.path.join(folder_path, audio_file)
                features = feature_extraction(audio_file_path)
                x_data.append(features)
                y_data.append(0 if class_name == 'Car' else 1)

    
    return np.array(x_data), np.array(y_data)

if __name__ == "__main__":

    # For building training dataset
    input_directory = "train_data_processed"
    x, y = build_dataset(input_directory)
    if not os.path.exists("train_dataset"):
        os.makedirs("train_dataset")
    else:
        os.system("rm -rf train_dataset/*") # Clear existing files in the dataset directory
    np.save("train_dataset/x_dataset.npy", x)
    np.save("train_dataset/y_dataset.npy", y)
    print(f"Saving training dataset with {x.shape[0]} samples.")


    # For building testing dataset
    input_directory = "test_data_processed"
    x, y = build_dataset(input_directory)
    if not os.path.exists("test_dataset"):
        os.makedirs("test_dataset")
    else:
        os.system("rm -rf test_dataset/*") # Clear existing files in the dataset directory
    np.save("test_dataset/x_dataset.npy", x)
    np.save("test_dataset/y_dataset.npy", y)
    print(f"Saving testing dataset with {x.shape[0]} samples.")


    # For building validation dataset
    input_directory = "val_data_processed"
    x, y = build_dataset(input_directory)
    if not os.path.exists("val_dataset"):
        os.makedirs("val_dataset")
    else:
        os.system("rm -rf val_dataset/*") # Clear existing files in the dataset directory
    np.save("val_dataset/x_dataset.npy", x)
    np.save("val_dataset/y_dataset.npy", y)
    print(f"Saving validation dataset with {x.shape[0]} samples.")