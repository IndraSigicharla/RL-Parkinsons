import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import silence_tensorflow.auto
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, roc_curve)
from tensorflow import constant
from tf_agents.trajectories import time_step


def network_predictions(network, X: np.ndarray) -> dict:
    q, _ = network(X, step_type=constant([time_step.StepType.FIRST] * X.shape[0]), training=False)
    return np.argmax(q.numpy(), axis=1)


def decision_function(network, X: np.ndarray) -> dict:
    q, _ = network(X, step_type=constant([time_step.StepType.FIRST] * X.shape[0]), training=False)
    return np.max(q.numpy(), axis=1)


def classification_metrics(y_true: list, y_pred: list) -> dict:
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = TP / (TP + FP) if TP + FP else 0  
    recall = TP / (TP + FN) if TP + FN else 0  
    specificity = TN / (TN + FP) if TN + FP else 0 

    G_mean = np.sqrt(recall * specificity)  
    F1 = f1_score(y_true, y_pred, zero_division=0)

    return {"G-Mean": G_mean, "F1": F1, "Precision": precision, "Recall": recall, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def plot_pr_curve(network, X_test: np.ndarray, y_test: np.ndarray, X_train: np.ndarray = None, y_train: np.ndarray = None) -> None: 
    plt.plot((0, 1), (1, 0), color="black", linestyle="--", label="Baseline")

    if X_train is not None and y_train is not None:
        y_val_score = decision_function(network, X_train)
        val_precision, val_recall, _ = precision_recall_curve(y_train, y_val_score)
        val_AP = average_precision_score(y_train, y_val_score)
        plt.plot(val_recall, val_precision, label=f"Train AP: {val_AP:.3f}")

    y_test_score = decision_function(network, X_test)
    test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_score)
    test_AP = average_precision_score(y_test, y_test_score)

    plt.plot(test_recall, test_precision, label=f"Test AP: {test_AP:.3f}")
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def plot_roc_curve(network, X_test: np.ndarray, y_test: np.ndarray, X_train: np.ndarray = None, y_train: np.ndarray = None) -> None:  
    plt.plot((0, 1), (0, 1), color="black", linestyle="--", label="Baseline")

    if X_train is not None and y_train is not None:
        y_train_score = decision_function(network, X_train)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_score)
        plt.plot(fpr_train, tpr_train, label=f"Train AUROC: {auc(fpr_train, tpr_train):.2f}")

    y_test_score = decision_function(network, X_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_score)

    plt.plot(fpr_test, tpr_test, label=f"Test AUROC: {auc(fpr_test, tpr_test):.2f}")
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(TP: int, FN: int, FP: int, TN: int) -> None: 
    sns.heatmap(((TP, FN), (FP, TN)), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

    plt.title("Confusion matrix")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()
