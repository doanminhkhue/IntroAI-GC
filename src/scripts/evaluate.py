# evaluate.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import get_data_generators
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix
import numpy as np

if __name__ == "__main__":
    _, _, test_gen = get_data_generators()
    model = load_model("model/model.h5")

    # Dự đoán
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print("Accuracy:", accuracy_score(y_true, y_pred) * 100)
    print("Precision:", precision_score(y_true, y_pred, average='weighted') * 100)
    print("Recall:", recall_score(y_true, y_pred, average='weighted') * 100)
    print("F1:", f1_score(y_true, y_pred, average='weighted') * 100)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
