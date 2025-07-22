import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score

# Generate synthetic data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=6, n_informative=4, random_state=42)

df = pd.DataFrame(X, columns=["income", "credit_score", "employment_length", "loan_amount", "dti", "self_employed"])
df["target"] = y

X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.2)

model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "ml/model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
