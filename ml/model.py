from sklearn.linear_model import LogisticRegression

# Demo training data (simulated)
X_train = [
    [20, 110, 15, 18, 115, 1],
    [28, 90, 13, 26, 95, 2],
    [32, 80, 12, 30, 85, 3],
]
y_train = [0, 1, 1]  # 0 = stable, 1 = escalating

model = LogisticRegression()
model.fit(X_train, y_train)

def predict_risk(features):
    probability = model.predict_proba([features])[0][1]
    return round(probability, 2)
