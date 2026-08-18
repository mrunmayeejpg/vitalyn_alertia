import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)

from train import generate_training_data, FEATURE_NAMES


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model():

    print("=" * 60)
    print("VITALYN ALERTIA - MODEL EVALUATION")
    print("=" * 60)


    # ========================================================
    # GENERATE TEST DATA
    # ========================================================

    df = generate_training_data()


    X = df[FEATURE_NAMES]

    y = df["risk"]


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    _, X_test, _, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y
    )


    print(
        f"\nTesting samples: {len(X_test)}"
    )


    # ========================================================
    # LOAD SAVED MODEL
    # ========================================================

    model_path = os.path.join(

        os.path.dirname(__file__),

        "saved_models",

        "sepsis_model.pkl"
    )


    if not os.path.exists(model_path):

        print(
            "\nERROR: Trained model not found."
        )

        print(
            "Run 'python ml/train.py' first."
        )

        return


    model = joblib.load(
        model_path
    )


    print(
        f"\nLoaded model:"
    )

    print(
        model
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = model.predict(
        X_test
    )


    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(

        y_test,

        predictions,

        zero_division=0
    )


    recall = recall_score(

        y_test,

        predictions,

        zero_division=0
    )


    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0
    )


    roc_auc = roc_auc_score(

        y_test,

        probabilities
    )


    # ========================================================
    # DISPLAY PERFORMANCE
    # ========================================================

    print(
        "\nMODEL PERFORMANCE"
    )

    print(
        "-" * 35
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        "-" * 35
    )

    print(
        classification_report(

            y_test,

            predictions,

            target_names=[
                "Lower Risk",
                "Higher Risk"
            ],

            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_test,

        predictions
    )


    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "-" * 35
    )

    print(
        cm
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_directory = os.path.join(

        os.path.dirname(__file__),

        "evaluation_results"
    )


    os.makedirs(

        output_directory,

        exist_ok=True
    )


    # ========================================================
    # CONFUSION MATRIX PLOT
    # ========================================================

    plt.figure(
        figsize=(7, 6)
    )


    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[
            "Lower Risk",
            "Higher Risk"
        ]
    )


    display.plot(
        cmap="Blues"
    )


    plt.title(
        "Vitalyn Alertia - Confusion Matrix"
    )


    plt.tight_layout()


    confusion_path = os.path.join(

        output_directory,

        "confusion_matrix.png"
    )


    plt.savefig(
        confusion_path,
        dpi=300
    )


    plt.close()


    print(
        f"\nConfusion matrix saved to:"
    )

    print(
        confusion_path
    )


    # ========================================================
    # ROC CURVE
    # ========================================================

    false_positive_rate, true_positive_rate, _ = roc_curve(

        y_test,

        probabilities
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.plot(

        false_positive_rate,

        true_positive_rate,

        linewidth=2,

        label=f"ROC-AUC = {roc_auc:.4f}"
    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        label="Random Classifier"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "Vitalyn Alertia - ROC Curve"
    )


    plt.legend()

    plt.grid(
        linestyle="--",
        alpha=0.4
    )


    plt.tight_layout()


    roc_path = os.path.join(

        output_directory,

        "roc_curve.png"
    )


    plt.savefig(

        roc_path,

        dpi=300
    )


    plt.close()


    print(
        f"\nROC curve saved to:"
    )

    print(
        roc_path
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score",

            "ROC-AUC"
        ],

        "Score": [

            accuracy,

            precision,

            recall,

            f1,

            roc_auc
        ]
    })


    metrics_path = os.path.join(

        output_directory,

        "evaluation_metrics.csv"
    )


    metrics.to_csv(

        metrics_path,

        index=False
    )


    print(
        f"\nEvaluation metrics saved to:"
    )

    print(
        metrics_path
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate_model()