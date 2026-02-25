import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


import dagshub
dagshub.init(repo_owner='ashokreddy304', repo_name='MLOPs_mlflow_1', mlflow=True)

# ----------------------------------------------------------
# 1. Configure MLflow Tracking & Experiment
# ----------------------------------------------------------
mlflow.set_tracking_uri("https://dagshub.com/ashokreddy304/MLOPs_mlflow_1.mlflow")
mlflow.set_experiment("wine-rf-experiment_2")

client = MlflowClient()

# Fetch experiment
exp = client.get_experiment_by_name("wine-rf-experiment_3")

# ----------------------------------------------------------
# 2. Add Experiment Description + Tags
# ----------------------------------------------------------
if exp is not None:

    # Add description (support all MLflow versions)
    client.set_experiment_tag(
        exp.experiment_id,
        "mlflow.experimentNote",
        "Random Forest hyperparameter tuning on Wine Quality dataset."
    )

    client.set_experiment_tag(
        exp.experiment_id,
        "mlflow.note.content",
        "Random Forest hyperparameter tuning on Wine Quality dataset."
    )

    # Add metadata tags
    client.set_experiment_tag(exp.experiment_id, "owner", "Ashok Reddy")
    client.set_experiment_tag(exp.experiment_id, "project", "wine-quality")
    client.set_experiment_tag(exp.experiment_id, "environment", "dev")

else:
    print("Experiment not found. Verify experiment name.")


# ----------------------------------------------------------
# 3. Load Dataset
# ----------------------------------------------------------
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model hyperparameters (actual values)
max_depth = 5
n_estimators = 10

# ----------------------------------------------------------
# 4. MLflow Run + Model Training
# ----------------------------------------------------------
with mlflow.start_run(run_name='V.0.0.1'):

    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_sco = f1_score(y_test, y_pred, average='macro')

    # ----------------------------------------------------------
    # Log parameters (corrected)
    # ----------------------------------------------------------
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("max_features", "sqrt")

    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("feature_engineering", "PCA")
    mlflow.log_param("pca_components", 8)

    mlflow.log_param("train_size", 0.7)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("data_version", "v1.0.3")

    # ----------------------------------------------------------
    # Log Metrics
    # ----------------------------------------------------------
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_sco)

    # ----------------------------------------------------------
    # Log Confusion Matrix Plot
    # ----------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=wine.target_names,
        yticklabels=wine.target_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save plot
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)

    # Log artifact
    mlflow.log_artifact(plot_path)

    # ----------------------------------------------------------
    # Log Python script file
    # ----------------------------------------------------------
    mlflow.log_artifact(__file__)

    #-----------------------------------------------------------
    # Add tags
    #-----------------------------------------------------------
    mlflow.set_tags({'Author':'Ashok',
                     "Project":'Wine Classification',
                     "created_by": "Ashok Reddy"})
    #-----------------------------------------------------------
    # Log Model
    #-----------------------------------------------------------
    mlflow.sklearn.log_model(rf,"RandomForest")

    print("Accuracy:", accuracy)

