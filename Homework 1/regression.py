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
    columns = ["m"+str(i) for i in range(0, m+1)]
    df = pd.DataFrame(columns= columns)
    for xi in x:
        df.loc[len(df)] = [xi ** i for i in range(0, m + 1)]
    return df

def trigonometric(x, m):
    columns = ["m" + str(i) for i in range(0, m + 1)]
    df = pd.DataFrame(columns=columns)
    for xi in x:
        df.loc[len(df)] =[ np.cos(2 * np.pi * xi * i) for i in range(0,  m)]
    return df

def schwartz_criterion(p, n):
    return 1 + p * ((1 - p) ** -1) * np.log(n)

def akaikes_fpe(p):
    return (1 + p) * ((1 - p) ** -1)

def plot(x, y, prediction, m):
    sorted_pairs = sorted(zip(x, prediction))
    sorted_x, sorted_y = zip(*sorted_pairs)

    plt.clf()
    plt.scatter(x, y)
    plt.scatter(x, prediction, color="b", label="Predictions")
    plt.plot(sorted_x, sorted_y, color="r", label="Regression Line")
    plt.title("Regression mit m: " + str(m))
    plt.show()

# %%
def linear_regression_model(df: pd.DataFrame, function):
    x = df.x
    y = df.y
    data_size = x.size
    schwartz = []
    coefs = []
    for m in range(1, data_size - 1):
        coefficients = function(x, m)
        lr = LinearRegression()
        lr.fit(coefficients, y)
        prediction = lr.predict(coefficients)
        mse = mean_squared_error(y, prediction)
        print("Mse: ",  mse)
        schwartz.append(schwartz_criterion(m / data_size, data_size) * mse)
        coefs.append(lr.coef_)
        # plot(x, y, prediction, m)

    min_m = np.argmin(schwartz)

    print("==============================================")
    print("Result for function:", function.__name__)
    print("Best m: " + str(min_m + 1))
    print("Estimation of: " + str(schwartz[min_m]))
    print("Coefs: " + str(coefs[min_m]))


if __name__ == '__main__':
    data = generate_data(20)
    linear_regression_model(data, algebraic)
    #linear_regression_model(data, trigonometric)

# Je höher die Sample size ist, desto eher wird sich dem Trend angenähert. Folglich sinkt komplexität
# Nein ist nicht möglich, da wir nur die besten Dimensionen des jeweiligen Modells zurückgegeben werden. Es findet jedoch keine Evaluierung darüber statt,
# welches der beiden Modelle besser ist. Ein Vergleich dieser wäre möglich, wird in unserer Modelselection jedoch nicht beachtet

