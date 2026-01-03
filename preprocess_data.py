import numpy as np
import librosa
import soundfile as sf
import os

def trim_wav_convertion_and_normalize(input_audio_path, output_audio_path, target_sr=16000):
    """
    Convert an audio file to 5 seconds, then to maximum amplitude 1.0, then convert to wav format and save it.

    Parameters:
    input_audio_path (str): Path to the input audio file.
    output_audio_path (str): Path to save the converted wav file.
    target_sr (int): Target sampling rate for the output file.

    """

    # Load the audio file with the target sampling rate
    audio_data, sr = librosa.load(input_audio_path, sr=target_sr)

    # Trim or pad the audio to 5 seconds
    target_length = target_sr * 5  # 5 seconds
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    else:
        padding = target_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')

    # Normalize the audio to have maximum amplitude of 1.0
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude > 0:
        audio_data = audio_data / max_amplitude

    # Save the normalized audio data as a wav file
    sf.write(output_audio_path, audio_data, target_sr)



def prepare_data(input_dir, output_dir, output_file_name, target_sr=16000):
    """
    Prepare audio data by converting all files in the input directory to normalized wav files in the output directory.
    Parameters:
    input_dir (str): Path to the input directory containing audio files.
    output_dir (str): Path to the output directory to save processed wav files.
    target_sr (int): Target sampling rate for the output files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        os.system(f"rm -rf {output_dir}/*") # Clear existing files in the output directory

    """
    Current dataset structure:
    input_dir/
        Class1/
            subfolder1/
                audiofile1
                audiofile2
            subfolder2/
                audiofile3
        Class2/
            subfolder1/
                audiofile4
                audiofile5
    Output dataset structure:
    output_dir/
        Class1/
            output_file_name_1.wav
            output_file_name_2.wav
        Class2/
            output_file_name_1.wav
            output_file_name_2.wav
    """


    for class_name in os.listdir(input_dir):
        index = 1
        folder_path = os.path.join(input_dir, class_name)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_dir, class_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            for folder_in_folder in os.listdir(folder_path):
                folder_in_folder_path =  os.path.join(folder_path, folder_in_folder)
                if os.path.isdir(folder_in_folder_path):
                    for file_name in os.listdir(folder_in_folder_path):
                        if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                            input_file_path = os.path.join(folder_in_folder_path, file_name)
                            output_file_name_full = f"{class_name}_{output_file_name}_{index}.wav"
                            output_file_path = os.path.join(output_folder_path, output_file_name_full)
                            trim_wav_convertion_and_normalize(input_file_path, output_file_path, target_sr)
                            index += 1
                            #print(f"Processed {input_file_path} -> {output_file_path}")
        print(f"Completed processing for class: {class_name}: {index - 1} files processed.")
                

if __name__ == "__main__":

    # For preparing training data
    input_directory = "train_data"
    output_directory = "train_data_processed"
    output_file_base_name = "processed_audio"
    prepare_data(input_directory, output_directory, output_file_base_name)

    # For preparing testing data
    input_directory = "test_data"
    output_directory = "test_data_processed"
    output_file_base_name = "processed_audio"
    prepare_data(input_directory, output_directory, output_file_base_name)

    # For preparing validation data
    input_directory = "val_data"
    output_directory = "val_data_processed"
    output_file_base_name = "processed_audio"
    prepare_data(input_directory, output_directory, output_file_base_name)