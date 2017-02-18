from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config as conf
import numpy as np
import pandas_datareader.data as pdr_data
import os
import sys
import tensorflow
import time


from collections import deque
from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error


"""
Copyright 2017 Daxiao(Shawn) Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def get_data():
    '''
    If filename exists, loads data, otherwise downloads and saves data from Yahoo Finance
    Returns:
    	- a list of arrays of daily values
    Credit to: Tencia Lee
    '''
    def download_data():
        from datetime import timedelta, datetime
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        print('Downloading data for dates {} - {}'.format(
            datetime.strftime(conf.start, "%Y-%m-%d"),
            datetime.strftime(conf.end, "%Y-%m-%d")))
        split = [0.8, 0.1, 0.1]

        cumusplit = [np.sum(split[:i]) for i,s in enumerate(split)]
        segment_start_dates = [conf.start + timedelta(
            days = int((conf.end - conf.start).days * interv)) for interv in cumusplit][::-1]
        stocks_list = map(lambda l: l.strip(), open(conf.names_file, 'r').readlines())
        by_stock = dict((s, pdr_data.DataReader(s, 'yahoo', conf.start, conf.end))
                for s in stocks_list)
        seq = [[],[],[]]
        for stock in by_stock:
            daily_returns = deque(maxlen=conf.normalize_std_len)
            for rec_date in (conf.start + timedelta(days=n) for n in xrange((conf.end-conf.start).days)):
                idx = next(i for i,d in enumerate(segment_start_dates) if rec_date >= d)
                try:
                    d = rec_date.strftime("%Y-%m-%d")
                    ac = by_stock[stock].ix[d]['Adj Close']
                    daily_return = ac / 1.0;
                    if len(daily_returns) == daily_returns.maxlen:
                        seq[idx].append(daily_return)
                    daily_returns.append(daily_return)
                except KeyError:
                    pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq][::-1]

    if not os.path.exists(conf.save_file):
        datasets = download_data()
        print('Saving in {}'.format(conf.save_file))
        np.savez(conf.save_file, *datasets)
    else:
	 with np.load(conf.save_file) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets


def sequence_iterator(raw_data, TIMESTEPS=2):
    """
       goal: turn data into x and y for feeding purpose
    """
    if TIMESTEPS <= 1:
	print("Invalid Timestep, timestep should be greater than 2")
	return
    input, target = [], []
    for i in xrange(len(raw_data) - TIMESTEPS - 1):
	x = raw_data[i:i+TIMESTEPS]
	y = raw_data[i+TIMESTEPS]
	input.append(x)
	target.append(y)
    return np.array(input), np.array(target)


train_data, valid_data, test_data = get_data()

train_x, train_y = sequence_iterator(train_data, conf.num_steps)
valid_x, valid_y = sequence_iterator(valid_data, conf.num_steps)
test_x, test_y   = sequence_iterator(test_data, conf.num_steps)

feature_columns = [tensorflow.contrib.layers.real_valued_column("", dimension=1)]

regressor = learn.DNNRegressor(
			feature_columns=feature_columns,
			hidden_units=[10, 10],
			optimizer=tensorflow.train.ProximalAdagradOptimizer(
      					learning_rate=0.03,
      					l1_regularization_strength=0.001
    				))

regressor.fit(train_x, train_y, steps=2000)

predicted = list(regressor.predict(test_x, as_iterable=True))
mae = mean_absolute_error(test_y, predicted)
print ("Error: %f" % mae)

#print('predictions: {}'.format(str(predicted)))

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
