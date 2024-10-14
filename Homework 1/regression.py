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

def algebraic(x, m):
    columns = ["m"+str(i) for i in range(1, m+1)]
    df = pd.DataFrame(columns= columns)
    for xi in x:
        df.loc[len(df)] = [xi ** i for i in range(1, m + 1)]
    return df

def trigonometric(x, m):
    columns = ["m" + str(i) for i in range(1, m + 1)]
    df = pd.DataFrame(columns=columns)
    for xi in x:
        df.loc[len(df)] =[ np.cos(2 * np.pi * xi * i) for i in range(1,  m+1)]
    return df


# %%
def linear_regression_model(df: pd.DataFrame):
    x = df.x
    y = df.y
    data_size = x.size

    m = 5
    #for m in range(1, data_size - 1):
    coefficients = algebraic(x, m)
    lr = LinearRegression()
    lr.fit(coefficients, y)
    prediction = lr.predict(coefficients)
    mse = mean_squared_error(y, prediction)

    sorted_pred = sorted(prediction)
    print(sorted_pred)

    plt.clf()
    plt.scatter(x, y)

    plt.scatter(x, prediction, color="b", label="Predictions")
    print(x)

    sorted_pairs = sorted(zip(x, prediction))
    sorted_x, sorted_y = zip(*sorted_pairs)

    # Linie durch die Vorhersagen ziehen
    plt.plot(sorted_x, sorted_y, color="r", label="Regression Line")

    plt.show()





if __name__ == '__main__':
    data = generate_data(100)
    linear_regression_model(data)
