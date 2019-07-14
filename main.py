import pandas as pd
import numpy as np
import os
import sys
import glob
import tarfile

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from fairing import PredictionEndpoint
from fairing import TrainJob
from fairing.backends import KubeflowBackend

from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists)

from data_model import StockDataSet

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 500, "Stock count [500]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("drop_prob", 0.2, "Drop probability of dropout layer. [0.2]")
flags.DEFINE_integer("num_train_steps", 1000, "Number of training steps. [1000]")
flags.DEFINE_float("test_ratio", 0.05, "Proportion of data to use for testing. [0.05]")
flags.DEFINE_string("model_dir_prefix", "models", "Directory to store model checkpoints and event files. [models]")

flags.DEFINE_boolean("training", True, "Whether to train or predict. [True]")
flags.DEFINE_boolean("local", True, "Whether to run locally [True]")

flags.DEFINE_string("endpoint", "minio-service:9000", "S3 endpoint [minio-service:9000]")
flags.DEFINE_string("access_key", None, "S3 access key")
flags.DEFINE_string("secret_key", None, "S3 secret key")

FLAGS = flags.FLAGS

#tf.enable_eager_execution()

STOCK_COUNT = FLAGS.stock_count
INPUT_SIZE = FLAGS.input_size
NUM_STEPS = FLAGS.num_steps
NUM_LAYERS = FLAGS.num_layers
LSTM_SIZE = FLAGS.lstm_size
BATCH_SIZE = FLAGS.batch_size
DROP_PROB = FLAGS.drop_prob
NUM_TRAIN_STEPS = FLAGS.num_train_steps
TEST_RATIO = FLAGS.test_ratio
MODEL_DIR_PREFIX = FLAGS.model_dir_prefix

TRAINING = FLAGS.training
LOCAL = FLAGS.local

ENDPOINT = FLAGS.endpoint
ACCESS_KEY = FLAGS.access_key
SECRET_KEY = FLAGS.secret_key

class FairingModel(object):

    def __init__(self):

        self.stock_count = STOCK_COUNT
        self.input_size = INPUT_SIZE
        self.num_steps = NUM_STEPS
        self.num_layers = NUM_LAYERS
        self.lstm_size = LSTM_SIZE
        self.batch_size = BATCH_SIZE
        self.drop_prob = DROP_PROB
        self.num_train_steps = NUM_TRAIN_STEPS
        self.test_ratio = TEST_RATIO
        self.model_dir_prefix = MODEL_DIR_PREFIX

        self.training = TRAINING
        self.local = LOCAL

        if not self.local:
            self.client = Minio(ENDPOINT,
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                secure=False)
            self.bucket_name = 'smp-rnn'

        inputs = layers.Input(shape=(self.num_steps,self.input_size))
        x = inputs
        for _ in range(self.num_layers - 1):
            # TODO: omit dropout when predicting
            x = layers.LSTM(self.lstm_size, dropout=self.drop_prob, recurrent_dropout=self.drop_prob, return_sequences=True)(x, training=self.training)
        outputs = layers.LSTM(self.lstm_size, dropout=self.drop_prob, recurrent_dropout=self.drop_prob)(x, training=self.training)

        keras_model = models.Model(inputs=inputs, outputs=outputs)
        keras_model.compile(optimizer='sgd',
            loss='mean_squared_error',
            metrics=['accuracy'])

        self.estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=self.model_dir)

    def load_stock_market_data(self):

        info = pd.read_csv("data/companylist.csv")
        info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
        info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
        #print(info['file_exists'].value_counts().to_dict())

        info = info[info['file_exists'] == True].reset_index(drop=True)
        info = info.sort_values('marketcap', ascending=False).reset_index(drop=True)

        if self.stock_count is not None:
            info = info.head(self.stock_count)

        self.stock_market_data = [
            StockDataSet(row['symbol'],
                        input_size=self.input_size,
                        num_steps=self.num_steps,
                        test_ratio=self.test_ratio)
            for _, row in info.iterrows()]

    def train_input_fn(self, features, labels, batch_size, repeat_count=None, perform_shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=100000)

        if repeat_count is not None:
            dataset = dataset.repeat(repeat_count)
        else:
            dataset = dataset.repeat()
            
        dataset = dataset.batch(batch_size) #, drop_remainder=True)

        return dataset

    def train(self):

        self.load_stock_market_data()

        self.features = {"input_1": np.concatenate([stock_data_set.train_data()[0] for stock_data_set in self.stock_market_data])}
        self.labels = np.concatenate([stock_data_set.train_data()[1] for stock_data_set in self.stock_market_data])

        self.estimator.train(input_fn=lambda:self.train_input_fn(self.features, self.labels, self.batch_size), steps=self.num_train_steps)

        self.save()

    def predict_input_fn(self, features):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features),))
        return dataset

    def predict(self, X, feature_names):

        self.restore()

        self.features = {"input_1": X}
        return str(list(self.estimator.predict(input_fn=lambda:self.predict_input_fn(self.features))))

        #prediction = self.model.predict(data=X)
        #return [[prediction.item(0), prediction.item(0)]]

        #return "{\"key\": \"value\"}"

    def save(self):

        if self.local:
            return

        try:
            self.client.make_bucket(self.bucket_name)
        except BucketAlreadyOwnedByYou as err:
            pass
        except BucketAlreadyExists as err:
            pass
        except ResponseError as err:
            raise

        archive_files = []
        cwd = os.getcwd()
        for (dirpath, dirnames, files) in os.walk(self.model_dir):
            for file_name in files:
                rel_dir = os.path.relpath(dirpath, cwd)
                rel_file = os.path.join(rel_dir, file_name)
                archive_files.append(rel_file)
            break

        archive = "%s.tar.gz" % self.model_name
        with tarfile.open(archive, "w:gz") as tar:
            for file_name in archive_files:
                tar.add(file_name)

        try:
            self.client.fput_object(self.bucket_name, archive, archive)
        except ResponseError as err:
            print(err)
            raise

    def restore(self):
        archive = "%s.tar.gz" % self.model_name

        try:
            self.client.fget_object(self.bucket_name, archive, archive)
        except ResponseError as err:
            print(err)
            raise

        with tarfile.open(archive) as tar:
            tar.extractall()

    @property
    def model_name(self):
        name = "stock_rnn_lstm%d_step%d_input%d" % (
            self.lstm_size, self.num_steps, self.input_size)

        return name

    @property
    def model_dir(self):
        model_dir = os.path.join(self.model_dir_prefix, self.model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

PY_VERSION = ".".join([str(x) for x in sys.version_info[0:3]])
TRAIN_BASE_IMAGE = 'python:{}'.format(PY_VERSION)

SELDON_CORE_PY_VERSION = "".join([str(x) for x in sys.version_info[0:2]])
PREDICT_BASE_IMAGE = 'seldonio/seldon-core-s2i-python{}:0.10'.format(PY_VERSION)

DOCKER_REGISTRY = 'dippynark'

def main(_):

    if TRAINING:
        if LOCAL:
            FairingModel().train()
        else:
            train_job = TrainJob(FairingModel,
                TRAIN_BASE_IMAGE,
                input_files=["data_model.py", "requirements.txt"] + glob.glob('data/*'),
                docker_registry=DOCKER_REGISTRY,
                backend=KubeflowBackend())
            train_job.submit()
    else:
        if LOCAL:
            pass
        else:
            endpoint = PredictionEndpoint(FairingModel,
                PREDICT_BASE_IMAGE,
                input_files=["data_model.py", "requirements.txt"],
                docker_registry=DOCKER_REGISTRY,
                backend=KubeflowBackend())
            endpoint.create()

if __name__ == '__main__':
    tf.app.run()