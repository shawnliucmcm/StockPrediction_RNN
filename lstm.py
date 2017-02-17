import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, load_csvdata, lstm_model


LOG_DIR = './ops_logs'
TIMESTEPS = 10
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)

feature_columns = [tensorflow.contrib.layers.real_valued_column("", dimension=1)]

X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)

regressor = learn.DNNRegressor(
			feature_columns=feature_columns,
			hidden_units=[10, 10],
			optimizer=tensorflow.train.ProximalAdagradOptimizer(
      					learning_rate=0.03,
      					l1_regularization_strength=0.001
    				))

regressor.fit(X['train'], y['train'], steps=2000)

predicted = list(regressor.predict(X['test'], as_iterable=True))

mae = mean_absolute_error(y['test'], predicted)
print ("Error: %f" % mae)

#print('predictions: {}'.format(str(predicted)))

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
