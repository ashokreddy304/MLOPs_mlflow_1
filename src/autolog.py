import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score,ConfusionMatrixDisplay
import matplotlib.pyplot as Plt
import seaborn as sns
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

mlflow.autolog()
mlflow.set_experiment("wine-rf-experiment_3")

#load_dataset

wine = load_wine()
x = wine.data
y = wine.target

#train test split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=42)

#define the params
max_depth = 5
n_estimators = 3
random_state = 0
with mlflow.start_run(run_name='RF_V.0.0.1'):
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=random_state)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_sco = f1_score(y_test, y_pred, average='macro')


    #creating confusion metric
    cm = confusion_matrix(y_test,y_pred)
    Plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    Plt.ylabel("Actual")
    Plt.xlabel("predicted")
    Plt.title("confusion metrics")

    #save plot
    Plt.savefig("confusion_metrix.png")

    #log artificats
    mlflow.log_artifact(__file__)

    print(accuracy)