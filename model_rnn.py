"""
@author: lilianweng
"""
import numpy as np
import os
import random
import re
import shutil
import time
import tarfile
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector

from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists)

def make_rnn_model(stock_count, dataset_list, batch_size, sample_size, max_epoch, init_learning_rate, learning_rate_decay, init_epoch, keep_prob,
    lstm_size=128, num_layers=1, num_steps=30, input_size=1, embed_size=None, logs_dir="logs", plots_dir="images", access_key=None, secret_key=None):
    class LstmRNN(object):
        def __init__(self, stock_count=stock_count,
                    dataset_list=dataset_list,
                    batch_size=batch_size,
                    sample_size=sample_size,
                    max_epoch=max_epoch,
                    init_learning_rate=init_learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    init_epoch=init_epoch,
                    keep_prob=keep_prob,
                    lstm_size=lstm_size,
                    num_layers=num_layers,
                    num_steps=num_steps,
                    input_size=input_size,
                    embed_size=embed_size,
                    logs_dir=logs_dir,
                    plots_dir=plots_dir):
            """
            Construct a RNN model using LSTM cell.

            Args:
                dataset_list (<StockDataSet>)
                config (tf.app.flags.FLAGS)
                stock_count (int): num. of stocks we are going to train with.
                lstm_size (int)
                num_layers (int): num. of LSTM cell layers.
                num_steps (int)
                input_size (int)
                keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
                embed_size (int): length of embedding vector, only used when stock_count > 1.
                checkpoint_dir (str)
            """

            self.client = Minio('minio-service:9000',
                  access_key=access_key,
                  secret_key=secret_key,
                  secure=False)
            self.bucket_name = 'smp-rnn'

            try:
                self.client.make_bucket(self.bucket_name)
            except BucketAlreadyOwnedByYou as err:
                pass
            except BucketAlreadyExists as err:
                pass
            except ResponseError as err:
                raise
            
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            run_config = tf.ConfigProto()
            run_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=run_config)

            self.stock_count = stock_count

            self.dataset_list = dataset_list
            self.batch_size=batch_size
            self.sample_size=sample_size
            self.max_epoch=max_epoch
            self.init_learning_rate=init_learning_rate
            self.learning_rate_decay=learning_rate_decay
            self.init_epoch=init_epoch
            self.keep_prob_value=keep_prob

            self.lstm_size = lstm_size
            self.num_layers = num_layers
            self.num_steps = num_steps
            self.input_size = input_size

            self.use_embed = (embed_size is not None) and (embed_size > 0)
            self.embed_size = embed_size or -1

            self.logs_dir = logs_dir
            self.plots_dir = plots_dir

            self.build_graph()

        def __del__(self):
            self.sess.close()

        def build_graph(self):
            """
            The model asks for five things to be trained:
            - learning_rate
            - keep_prob: 1 - dropout rate
            - symbols: a list of stock symbols associated with each sample
            - input: training data X
            - targets: training label y
            """
            # inputs.shape = (number of examples, number of input, dimension of each input).
            self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
            self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

            # Stock symbols are mapped to integers.
            self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

            self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
            self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

            def _create_one_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
                return lstm_cell

            cell = tf.contrib.rnn.MultiRNNCell(
                [_create_one_cell() for _ in range(self.num_layers)],
                state_is_tuple=True
            ) if self.num_layers > 1 else _create_one_cell()

            if self.embed_size > 0 and self.stock_count > 1:
                self.embed_matrix = tf.Variable(
                    tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                    name="embed_matrix"
                )

                # stock_label_embeds.shape = (batch_size, embedding_size)
                stacked_symbols = tf.tile(self.symbols, [1, self.num_steps], name='stacked_stock_labels')
                stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix, stacked_symbols)

                # After concat, inputs.shape = (batch_size, num_steps, input_size + embed_size)
                self.inputs_with_embed = tf.concat([self.inputs, stacked_embeds], axis=2, name="inputs_with_embed")
                self.embed_matrix_summ = tf.summary.histogram("embed_matrix", self.embed_matrix)

            else:
                self.inputs_with_embed = tf.identity(self.inputs)
                self.embed_matrix_summ = None

            print("inputs.shape:", self.inputs.shape)
            print("inputs_with_embed.shape:", self.inputs_with_embed.shape)

            # Run dynamic RNN
            val, state_ = tf.nn.dynamic_rnn(cell, self.inputs_with_embed, dtype=tf.float32, scope="dynamic_rnn")

            # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
            # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
            val = tf.transpose(val, [1, 0, 2])

            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
            ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
            bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
            self.pred = tf.matmul(last, ws) + bias

            self.last_sum = tf.summary.histogram("lstm_state", last)
            self.w_sum = tf.summary.histogram("w", ws)
            self.b_sum = tf.summary.histogram("b", bias)
            self.pred_summ = tf.summary.histogram("pred", self.pred)

            # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
            self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

            # Separated from train loss.
            self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

            self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
            self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
            self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

            self.t_vars = tf.trainable_variables()
            self.saver = tf.train.Saver()

        def train(self):
            assert len(self.dataset_list) > 0
            self.merged_sum = tf.summary.merge_all()

            # Set up the logs folder
            self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
            self.writer.add_graph(self.sess.graph)

            if self.use_embed:
                # Set up embedding visualization
                # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
                projector_config = projector.ProjectorConfig()

                # You can add multiple embeddings. Here we add only one.
                added_embed = projector_config.embeddings.add()
                added_embed.tensor_name = self.embed_matrix.name
                # Link this tensor to its metadata file (e.g. labels).
                shutil.copyfile(os.path.join(self.logs_dir, "metadata.tsv"),
                                os.path.join(self.model_logs_dir, "metadata.tsv"))
                added_embed.metadata_path = "metadata.tsv"

                # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
                # read this file during startup.
                projector.visualize_embeddings(self.writer, projector_config)

            tf.global_variables_initializer().run(session=self.sess)

            # Merged test data of different stocks.
            merged_test_X = []
            merged_test_y = []
            merged_test_labels = []

            for label_, d_ in enumerate(self.dataset_list):
                merged_test_X += list(d_.test_X)
                merged_test_y += list(d_.test_y)
                merged_test_labels += [[label_]] * len(d_.test_X)

            merged_test_X = np.array(merged_test_X)
            merged_test_y = np.array(merged_test_y)
            merged_test_labels = np.array(merged_test_labels)

            print("len(merged_test_X) =", len(merged_test_X))
            print("len(merged_test_y) =", len(merged_test_y))
            print("len(merged_test_labels) =", len(merged_test_labels))

            test_data_feed = {
                self.learning_rate: 0.0,
                self.keep_prob: 1.0,
                self.inputs: merged_test_X,
                self.targets: merged_test_y,
                self.symbols: merged_test_labels,
            }

            global_step = 0

            num_batches = sum(len(d_.train_X) for d_ in self.dataset_list) // self.batch_size
            random.seed(time.time())

            # Select samples for plotting.
            sample_labels = range(min(self.sample_size, len(self.dataset_list)))
            sample_indices = {}
            for l in sample_labels:
                sym = self.dataset_list[l].stock_sym
                target_indices = np.array([
                    i for i, sym_label in enumerate(merged_test_labels)
                    if sym_label[0] == l])
                sample_indices[sym] = target_indices
            print(sample_indices)

            print("Start training for stocks:", [d.stock_sym for d in self.dataset_list])
            for epoch in range(self.max_epoch):
                epoch_step = 0
                learning_rate = self.init_learning_rate * (
                    self.learning_rate_decay ** max(float(epoch + 1 - self.init_epoch), 0.0)
                )

                for label_, d_ in enumerate(self.dataset_list):
                    for batch_X, batch_y in d_.generate_one_epoch(self.batch_size):
                        global_step += 1
                        epoch_step += 1
                        batch_labels = np.array([[label_]] * len(batch_X))
                        train_data_feed = {
                            self.learning_rate: learning_rate,
                            self.keep_prob: self.keep_prob_value,
                            self.inputs: batch_X,
                            self.targets: batch_y,
                            self.symbols: batch_labels,
                        }
                        train_loss, _, train_merged_sum = self.sess.run(
                            [self.loss, self.optim, self.merged_sum], train_data_feed)
                        self.writer.add_summary(train_merged_sum, global_step=global_step)

                        if np.mod(global_step, len(self.dataset_list) * 200 / self.input_size) == 1:
                            test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)

                            print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                                global_step, epoch, learning_rate, train_loss, test_loss))

                            # Plot samples
                            for sample_sym, indices in sample_indices.items():
                                image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                                    sample_sym, epoch, epoch_step))
                                sample_preds = test_pred[indices]
                                sample_truth = merged_test_y[indices]
                                self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)

                            self.save(global_step)

            final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

            # Save the final model
            self.save(global_step)

            self.upload()

        @property
        def model_name(self):
            name = "stock_rnn_lstm%d_step%d_input%d" % (
                self.lstm_size, self.num_steps, self.input_size)

            if self.embed_size > 0:
                name += "_embed%d" % self.embed_size

            return name

        @property
        def model_logs_dir(self):
            model_logs_dir = os.path.join(self.logs_dir, self.model_name)
            if not os.path.exists(model_logs_dir):
                os.makedirs(model_logs_dir)
            return model_logs_dir

        @property
        def model_plots_dir(self):
            model_plots_dir = os.path.join(self.plots_dir, self.model_name)
            if not os.path.exists(model_plots_dir):
                os.makedirs(model_plots_dir)
            return model_plots_dir

        def save(self, step):
            model_name = self.model_name + ".model"
            save_path = os.path.join(self.model_logs_dir, model_name)
            self.saver.save(
                self.sess,
                save_path,
                global_step=step
            )

        def upload(self):
            archive_files = []
            cwd = os.getcwd()
            for (dirpath, dirnames, files) in os.walk(self.model_logs_dir):
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

        def load(self):
            print(" [*] Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter

            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0

        def plot_samples(self, preds, targets, figname, stock_sym=None, multiplier=5):
            def _flatten(seq):
                return np.array([x for y in seq for x in y])

            truths = _flatten(targets)[-200:]
            preds = (_flatten(preds) * multiplier)[-200:]
            days = range(len(truths))[-200:]

            plt.figure(figsize=(12, 6))
            plt.plot(days, truths, label='truth')
            plt.plot(days, preds, label='pred')
            plt.legend(loc='upper left', frameon=False)
            plt.xlabel("day")
            plt.ylabel("normalized price")
            plt.ylim((min(truths), max(truths)))
            plt.grid(ls='--')

            if stock_sym:
                plt.title(stock_sym + " | Last %d days in test" % len(truths))

            plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
            plt.close()
    return LstmRNN
