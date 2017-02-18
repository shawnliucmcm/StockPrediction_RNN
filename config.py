import datetime

names_file = 'data/stock.csv'

start = datetime.datetime(2005, 1, 1)
end = datetime.datetime(2017, 2, 14)

save_file = 'data/stock_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
normalize_std_len = 50
num_steps = 10


class Config(object):
    """configuration for RNN."""
    init_scale = 0.1
    learning_rate = 0.03
    max_grad_norm = 5
    num_layers = 6
    num_steps = 10
    hidden_size = 200
    keep_prob = 1
    lr_decay = 0.5
    batch_size = 10
