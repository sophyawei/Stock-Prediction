import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def get_data(filename):
    my_csv = pd.read_csv(filename, usecols=['Date', 'Adj Close'])
    return my_csv

def predict_price_linear_regression(my_csv):
    #reshape index column to 2D array for .fit() method
    x_train = np.array(my_csv.index).reshape(-1, 1)
    y_train = my_csv['Adj Close']

    #create LinearRegression object
    model = linear_model.LinearRegression()
    #fit linear model by using the train dataset
    model.fit(x_train, y_train)

    #model evaluation,
    #thanks for https://github.com/mediasittich/Predicting-Stock-Prices-with-Linear-Regression
    #slope coefficient tells us that with a 1 unit increase in date the closing price increases by how much
    print('Slope: ', np.squeeze(model.coef_).item())
    #intercept coefficient is the price at which the closing price measurement started, the stock price value at date zero
    print('Intercept: ', model.intercept_)

    #train set graph
    plt.title('Linear Regression')
    plt.scatter(x_train, y_train, edgecolor='w', label='Actual Price')
    plt.plot(x_train, model.predict(x_train), color= 'red', label= 'Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Adjusted closing price $')
    plt.legend()
    plt.show()
    return

my_csv = get_data('TGODF.csv')
predict_price_linear_regression(my_csv)
