import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


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

def schwartz_criterion(p, n):
    return 1 + p * ((1 - p) ** -1) * np.log(n)

def plot(x, y, prediction, m):
    sorted_pairs = sorted(zip(x, prediction))
    sorted_x, sorted_y = zip(*sorted_pairs)

    plt.clf()
    plt.scatter(x, y)
    plt.scatter(x, prediction, color="b", label="Predictions")
    plt.plot(sorted_x, sorted_y, color="r", label="Regression Line")
    plt.title("Regression mit m: " + str(m))
    plt.show()

def akaikes_fpe(p):
    return (1 + p) * ((1 - p) ** -1)
# %%
def linear_regression_model(df: pd.DataFrame, function):
    x = df.x
    y = df.y
    data_size = x.size
    schwartz = []
    kfold = KFold(shuffle=True)
    for m in range(1, data_size - 1):
        kfold_result = []
        for train_split, test_split in kfold.split(x, y):
            train_x = np.array([x[i] for i in train_split])
            train_y = np.array([y[i] for i in train_split])
            test_x = np.array([[x[i]] for i in test_split])
            test_y = np.array([y[i] for i in test_split])

            coefficients_train = function(train_x, m)
            coefficients_test = function(test_x, m)

            lr = LinearRegression()
            lr.fit(coefficients_train, train_y)
            prediction = lr.predict(coefficients_test)

            # print("--------------------------------")
            # print("test - y: ", test_y)
            # print("Prediction: ", prediction)

            mse = mean_squared_error(test_y, prediction)
            print("Mse: ", mse)
            kfold_result.append(schwartz_criterion(m / data_size, data_size) * mse)

        schwartz.append(np.mean(kfold_result))

    min_m = np.argmin(schwartz)
    print("Schwartz: ", schwartz)

    print("==============================================")
    print("Result for function:", function.__name__)
    print("Best m: " + str(min_m + 1))
    print("Estimation of: " + str(schwartz[min_m]))


if __name__ == '__main__':
    data = generate_data(100)
    linear_regression_model(data, algebraic)
    linear_regression_model(data, trigonometric)

# Durch hohe m's haben wir extremes overfitting. es gibt daher immer extreme ausreißer. Das hat zur Folge, dass kleine m das ganze besser darstellen.