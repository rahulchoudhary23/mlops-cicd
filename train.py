# train.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# save model
joblib.dump(model, "app/model.pkl")

print("Model trained and saved!")
