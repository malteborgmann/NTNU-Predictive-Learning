import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_data(n=10):
    x = np.random.uniform(0, 1, n)
    noise = np.random.normal(0, 0.5, n)
    y = x**2 + 0.1 * x + noise
    return pd.DataFrame({'x': x, 'y': y})

def algebraic_model(x, m):
    w = []
    for xi in x:
        w = [x ** i for i in range(1, m + 1)]
    return pd.DataFrame(w)

def trigonometric_model(x, m):
    w = []
    for xi in x:
        w = [ np.cos(2 * np.pi * x * i) for i in range(1,  m+1)]
    return pd.DataFrame(w)

def schwartz_criterion(p, n):
    return 1 + p * ((1 - p) ** -1) * np.log(n)

def linear_regression_model(df: pd.DataFrame, estimations: pd.DataFrame):
    data_size = df['x'].size
    transpose = estimations.transpose()
    schwartz = []
    for m in range(1, data_size - 1):
        lr = LinearRegression()
        lr.fit(transpose.iloc[:, 0:m].values.tolist(), df['y'])
        prediction = lr.predict(transpose.iloc[:, 0:m].values.tolist())
        mse = mean_squared_error(df['y'], prediction)
        schwartz.append(schwartz_criterion(m / data_size, data_size) * mse)
    return schwartz

data = generate_data()
print(data)
data_size = data.x.size
print(data_size)
trigonometric_estimations = trigonometric_model(data.x, data_size)
algebraic_estimations = algebraic_model(data.x, data_size)


schwartz_trigonometric = linear_regression_model(data, trigonometric_estimations)
schwartz_algebraic = linear_regression_model(data, trigonometric_estimations)

