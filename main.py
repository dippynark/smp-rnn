import os
import pandas as pd
import pprint
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

from fairing import TrainJob
from fairing.backends import KubeflowBackend

from data_model import StockDataSet
import model_rnn

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

flags.DEFINE_string("access_key", None, "S3 access key")
flags.DEFINE_string("secret_key", None, "S3 secret key")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if not os.path.exists("logs"):
    os.mkdir("logs")

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_stock_data(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]

    # Load stocks metadata
    info = pd.read_csv("data/companylist.csv")
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print(info['file_exists'].value_counts().to_dict())

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort_values('marketcap', ascending=False).reset_index(drop=True)

    if k is not None:
        info = info.head(k)

    print("Stocks info:\n", info.head())

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataSet(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05)
        for _, row in info.iterrows()]

PY_VERSION = ".".join([str(x) for x in sys.version_info[0:3]])
BASE_IMAGE = 'python:{}'.format(PY_VERSION)
DOCKER_REGISTRY = 'dippynark'

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    #show_all_variables()

    stock_data_list = load_stock_data(
        FLAGS.input_size,
        FLAGS.num_steps,
        k=FLAGS.stock_count,
        target_symbol=FLAGS.stock_symbol,
    )

    print(tf.app.flags.FLAGS.flag_values_dict())

    rnn_model = model_rnn.make_rnn_model(FLAGS.stock_count,
        stock_data_list,
        FLAGS.batch_size,
        FLAGS.sample_size,
        FLAGS.max_epoch,
        FLAGS.init_learning_rate,
        FLAGS.learning_rate_decay,
        FLAGS.init_epoch,
        FLAGS.keep_prob,
        lstm_size=FLAGS.lstm_size,
        num_layers=FLAGS.num_layers,
        num_steps=FLAGS.num_steps,
        input_size=FLAGS.input_size,
        embed_size=FLAGS.embed_size,
        access_key=FLAGS.access_key,
        secret_key=FLAGS.secret_key,
        )

    train_job = TrainJob(rnn_model, BASE_IMAGE, input_files=["data_model.py", "requirements.txt", "logs/metadata.tsv"], docker_registry=DOCKER_REGISTRY, backend=KubeflowBackend())
    train_job.submit()

    #if FLAGS.train:
    #    rnn_model().train(stock_data_list, FLAGS)
    #else:
    #    if not rnn_model.load()[0]:
    #        raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
    tf.app.run()