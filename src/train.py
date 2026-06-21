import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import random
import numpy as np

# Set random seed
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("data/iris.csv")

# Features and target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Train model
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

# Save metrics
metrics = pd.DataFrame({
    "Model": ["DummyClassifier"],
    "Accuracy": [accuracy]
})

metrics.to_csv("logs/metrics.csv", index=False)

print("Metrics saved successfully!")