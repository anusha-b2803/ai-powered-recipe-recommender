import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_classification_metrics(model, X_test, y_test):
    """
    Computes accuracy, precision, recall, F1, and prints classification reports.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Classification Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return acc, report, cm

def get_cross_val_score(model, X, y, cv=5):
    """
    Runs cross-validation score over model pipeline.
    """
    print("\n--- Running Cross Validation ---")
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.2f})")
    
    return scores

def plot_confusion_matrix(cm, labels, save_dir=None):
    """
    Plots and saves confusion matrix heatmap using seaborn.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(path)
        print(f"Confusion Matrix saved to {path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Test stub
    pass
