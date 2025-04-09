import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)

true_slope = 1.5
true_intercept = 2
noise = np.random.normal(0, 1, size=X.shape[0])
y = (true_slope * X.flatten()) + true_intercept + noise

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

slope = model.coef_[0]
intercept = model.intercept_
equation = f"y = {slope:.2f}x + {intercept:.2f}"

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=10, label='Données avec bruit')
plt.plot(X, y_pred, color='red', linewidth=2, label='Régression linéaire')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression Linéaire sur 100 Points')
plt.legend()
plt.grid(True)
plt.text(0.5, max(y) - 2, equation, fontsize=12, color='red')
plt.show()
