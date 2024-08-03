import mlflow
import mlflow.sklearn
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=args.random_state)

# Train the RandomForest classifier
clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log parameters and metrics
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_param("random_state", args.random_state)
mlflow.log_metric("accuracy", accuracy)

# Log the model
mlflow.sklearn.log_model(clf, "model")

# Output result (optional)
print(f"Model accuracy: {accuracy}")
