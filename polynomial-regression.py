import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)

true_coefficients = [1.2, -0.8, 3]  # y = 1.2x² - 0.8x + 3
noise = np.random.normal(0, 2, size=X.shape[0])
y = (true_coefficients[0] * X.flatten()**2 +
     true_coefficients[1] * X.flatten() +
     true_coefficients[2] + noise)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

coef = model.coef_

intercept = model.intercept_
equation = f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {intercept:.2f}"

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=10, label='Données avec bruit')
plt.plot(X, y_pred, color='green', linewidth=2, label='Régression polynomiale')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression Polynomiale (degré 2)')
plt.legend()
plt.grid(True)
plt.text(1, max(y) - 5, equation, fontsize=12, color='green')
plt.show()
