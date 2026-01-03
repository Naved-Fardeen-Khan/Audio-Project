import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import re




if __name__ == "__main__":
    # Load trained SVM model
    svm_model = joblib.load("models/svm_model_50_samples.pkl")
    print("Loaded SVM model from models/svm_model_50_samples.pkl") if svm_model else print("Failed to load SVM model.")

    # Load validation dataset
    input_directory = "val_dataset"
    x_val = np.load(os.path.join(input_directory, "x_dataset.npy"))
    y_val = np.load(os.path.join(input_directory, "y_dataset.npy"))
    print(f"Validation dataset loaded with {x_val.shape[0]} samples.")

    # Make predictions
    y_pred = svm_model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_val, y_pred)
    class_report = classification_report(y_val, y_pred, zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Plot confusion matrix as heatmap
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Greens',
                xticklabels=['Car', 'Tram'],
                yticklabels=['Car', 'Tram'],
                square=True,
                )    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap for 25 samples per class')
    plt.savefig("confusion_matrix_heatmap_25_samples.png")
    plt.show()
    plt.close()

    # Accuracy vs Number of Samples Plot
    sample_sizes = []
    accuracies = []

    # Regex to extract number of samples from filename
    pattern = r"svm_model_(\d+)_samples\.pkl"
    model_dir = "models"
    for model_file in os.listdir(model_dir):
        match = re.match(pattern, model_file)
        if match:
            num_samples = int(match.group(1))
            sample_sizes.append(num_samples)

            # Load model
            model_path = os.path.join(model_dir, model_file)
            model = joblib.load(model_path)

            # Make predictions
            y_pred_model = model.predict(x_val)
            acc = accuracy_score(y_val, y_pred_model)
            accuracies.append(acc)
            print(f"Model: {model_file}, Samples per class: {num_samples//2}, Accuracy: {acc:.4f}")

    # Sort by sample sizes
    sorted_indices = np.argsort(sample_sizes)
    sample_sizes = np.array(sample_sizes)[sorted_indices]
    accuracies = np.array(accuracies)[sorted_indices]

    # Plot
    os.makedirs("results", exist_ok=True)
    plt.plot(sample_sizes // 2, accuracies, marker='.')
    plt.xlabel('Number of Samples per Class')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Number of Samples per Class')
    plt.grid()
    plt.savefig("results/accuracy_vs_samples_per_class.png")
    plt.show()
    plt.close()
