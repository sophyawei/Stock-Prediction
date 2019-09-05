import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def get_data(filename):
    my_csv = pd.read_csv(filename, usecols=['Date', 'Adj Close'])
    return my_csv

def predict_price_svr(my_csv):
    #reshape index column to 2D array for .fit() method
    x_train = np.array(my_csv.index).reshape(-1, 1)
    y_train = my_csv['Adj Close']

    #increase SVM speed
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
    x_train = scaling.transform(x_train)

    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    svr_lin = SVR(kernel= 'linear', C= 1e3, gamma = 'auto')
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 3, gamma = 'auto')

    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF model', 'Linear model', 'Polynomial model']
    model_color = ['m', 'c', 'g']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharey=True)
    for i, svr in enumerate(svrs):
        axes[i].plot(x_train, svr.fit(x_train, y_train).predict(x_train), color=model_color[i], label=kernel_label[i])
        axes[i].scatter(x_train, y_train, edgecolor=model_color[i], label='Actual Price')
        axes[i].legend()
    fig.suptitle("Support Vector Regression")
    plt.show()
    return

my_csv = get_data('TGODF.csv')
predict_price_svr(my_csv)
