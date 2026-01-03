import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

def check_dataset_consistency(x, y):
    """
    Check if the feature dataset and label dataset have consistent number of samples.

    Parameters:
    x (np.ndarray): Feature dataset.
    y (np.ndarray): Label dataset.

    Returns:
    bool: True if datasets are consistent, False otherwise.
    """
    if x.shape[0] != y.shape[0]:
        print(f"Inconsistent dataset sizes: x has {x.shape[0]} samples, y has {y.shape[0]} samples.")
        return False
    print("Datasets are consistent. Both have", x.shape[0], "samples.")
    return True



if __name__ == "__main__":
    # Load training dataset
    input_directory = "train_dataset"
    x_train = np.load(os.path.join(input_directory, "x_dataset.npy"))
    y_train = np.load(os.path.join(input_directory, "y_dataset.npy"))
    if not check_dataset_consistency(x_train, y_train):
        exit(1)

    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        os.system(f"rm -rf {output_dir}/*") # Clear existing files in the output directory
    
    car_idx = np.where(y_train == 0)[0]
    tram_idx = np.where(y_train == 1)[0]
    print(f"Training dataset contains {len(car_idx)} 'Car' samples and {len(tram_idx)} 'Tram' samples.")


    # Select equal number of samples from both classes for balanced training
    # Increasing the number of samples used to train by multiples of 1
    for sample_size in range(1, min(len(car_idx), len(tram_idx)) + 1, 1):
        selected_car_idx = car_idx[:sample_size]
        selected_tram_idx = tram_idx[:sample_size]
        selected_indices = np.concatenate((selected_car_idx, selected_tram_idx))
        np.random.shuffle(selected_indices)

        x_subset = x_train[selected_indices]
        y_subset = y_train[selected_indices]

        print(f"Training SVM model with {sample_size} samples from each class ({2 * sample_size} total samples).")

        #Train SVM model
        svm_model = make_pipeline(StandardScaler(),
                                SVC(kernel='rbf',
                                    C=1.0,
                                    gamma='scale',
                                    probability=True,
                                    random_state=77) # Why 77? I like the number
                                )
        svm_model.fit(x_subset, y_subset)
        model_filename = os.path.join(output_dir, f"svm_model_{2 * sample_size}_samples.pkl")
        joblib.dump(svm_model, model_filename) # Save the trained model
        print(f"SVM model saved to {model_filename}")

        if sample_size == 30:
            print("Reached maximum balanced sample size for training.")
            break


    # Train SVM model
    svm_model = make_pipeline(StandardScaler(),
                            SVC(kernel='rbf',
                                C=1.0,
                                gamma='scale',
                                probability=True,
                                random_state=77)
                            )
    svm_model.fit(x_train, y_train)
    joblib.dump(svm_model, "models/svm_model_final.pkl") # Save the trained model
    print("SVM model saved to svm_model_final.pkl")