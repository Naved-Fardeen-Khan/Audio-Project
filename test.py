import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load trained SVM model
    svm_model = joblib.load("models/svm_model_final.pkl")
    print("Loaded SVM model from models/svm_model_final.pkl") if svm_model else print("Failed to load SVM model.")

    # Load test dataset
    input_directory = "test_dataset"
    x_test = np.load(os.path.join(input_directory, "x_dataset.npy"))
    y_test = np.load(os.path.join(input_directory, "y_dataset.npy"))
    print(f"Test dataset loaded with {x_test.shape[0]} samples.")

    # Make predictions
    y_pred = svm_model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Plot confusion matrix as heatmap
    os.makedirs("results", exist_ok=True)
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
    plt.title('Confusion Matrix Heatmap')
    plt.savefig("results/test_confusion_matrix_heatmap.png")
    plt.show()
    plt.close()