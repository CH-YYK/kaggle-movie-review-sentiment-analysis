import tensorflow as tf
import numpy as np
import pandas as pd
import re

#train_path = '../input/train.tsv'
#test_path = '../input/test.tsv'
#word_vec = '../input/glove6b/glove.6B.200d.txt'


train_path = 'data/train.tsv'
test_path = 'data/test.tsv'
word_vec = 'data/glove.6B.100d.txt'

# ---- load training dataset ----
class data_tool(object):

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        # training set
        self.train = pd.read_table(train_path, sep="\t")
        self.train_x = [self.str_clean(i) for i in self.train['Phrase']]

        self.train_y = [[0] * 4 for i in range(self.train.shape[0])]
        _ = [self.train_y[i].insert(j, 1) for i, j in enumerate(self.train['Sentiment'])]
        self.train_y = np.array(self.train_y)

        # test set
        self.test = pd.read_table(test_path, sep='\t')
        self.test_x = [self.str_clean(i) for i in self.test['Phrase']]

        # build corpus
        self.max_length, self.vocab_dict = self.word_corpus(self.train_x + self.test_x)

        # Tokenize, convert text to a list of integers
        self.train_x = self.text2index(self.train_x, self.vocab_dict, self.max_length)
        self.test_x = self.text2index(self.test_x, self.vocab_dict, self.max_length)

    def word_corpus(self, text_x):
        # Tokenize words
        vocab_dict = {word: index + 1 for index, word in enumerate(set(' '.join(text_x).split()))}
        vocab_dict['<UNK>'] = 0

        # maximum sequence length
        max_sequence_length = len(max([i.split() for i in text_x], key=len))
        return max_sequence_length, vocab_dict

    def text2index(self, text, vocab_dict, maximum_length):
        """
        tokenization
        """
        text = [i.split() for i in text]
        tmp = np.zeros(shape=(len(text), maximum_length))
        for i in range(len(text)):
            for j in range(len(text[i])):
                tmp[i][j] = vocab_dict.get(text[i][j], 0)
        return tmp

    def str_clean(self, string):
        """
        Tokenization/string cleaning forn all dataset except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        return string
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    # generate batches of data to train
    def generate_batches(self, data, epoch_size, batch_size, shuffle=False):
        data = np.array(data)

        data_size = len(data)

        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j+1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def save_data(self, result):
        test_data = pd.read_table(test_path, sep='\t')
        test_data['Sentiment'] = result.reshape(-1).tolist()
        test_data = test_data.loc[:, ['PhraseId', 'Sentiment']]
        print(test_data)
        test_data.to_csv("sample_submission.csv", index=False)


# --- build RNN model ----
class TextRNN(object):

    def __init__(self, sequence_len, embedding_size, num_classes, vocabulary_size, LSTM_size,
                 cnn_filer_size, num_filters, max_pooling_size, word_vec=None):

        # define params
        self.embedding_size = embedding_size
        self.sequence_length = sequence_len
        self.max_pooling_size = max_pooling_size

        # define placeholders
        self.input_x = tf.placeholder(tf.int32, [None, sequence_len], name='input_x')
        self.label_y = tf.placeholder(tf.int32, [None, num_classes], name='label_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.padding = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='padding')
        self.real_seq_length = tf.placeholder(tf.int32, [None], name='real_seq_length')

        # character embedding
        if word_vec is not None:
            with tf.name_scope('character_embedding'):
                W = tf.get_variable("embedding_weight", initializer=tf.constant(word_vec, dtype=tf.float32),
                                    trainable=False)
                self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_char')
                self.embedded_char_expand = tf.expand_dims(self.embedded_char, axis=-1)
        else:
            with tf.device("/cpu:0"), tf.name_scope('character_embedding'):
                W = tf.get_variable("embedding_weight", initializer=tf.truncated_normal([vocabulary_size, embedding_size],
                                                                                        mean=0, stddev=0.05))
                self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_char')
                self.embedded_char_expand = tf.expand_dims(self.embedded_char, axis=-1)


        # cnn
        with tf.name_scope('conv_nets'):
            pooling_output = []
            for index, filter_size in enumerate(cnn_filer_size):
                # padding embedding charaters to make sure that sequences will be equal after convnets
                num_prio = (filter_size - 1) // 2
                num_post = filter_size-1 - num_prio
                embedding_pad = tf.concat([self.padding]*num_prio + [self.embedded_char_expand] + [self.padding]*num_post,
                                          axis=1)

                # apply pooling convnets to padded chars_embedded
                pooling = self.cnn(embedding_pad, filter_size, index, embedding_size, num_filters[index])
                pooling_output.append(pooling)

            # concating all outputs into
            self.pool_output = tf.concat(pooling_output, axis=-1)

        # Highway
        # with tf.name_scope('Highway'):
        #     for i in range(num_Highway):
        #         self.pool_output = self.highway(self.pool_output, activation=tf.nn.relu)

        # LSTM
        with tf.name_scope('LSTM_nets'):
            reduced_length = self.real_seq_length // max_pooling_size
            self.LSTM_cell = tf.contrib.rnn.BasicRNNCell(LSTM_size)
            self.LSTM_cell = tf.contrib.rnn.DropoutWrapper(self.LSTM_cell, output_keep_prob=self.keep_prob)
            self.outputs, self.states = tf.nn.dynamic_rnn(self.LSTM_cell, self.pool_output,
                                                          sequence_length=reduced_length,
                                                          dtype=tf.float32)

        # Fully_connected and Dropout
        with tf.name_scope("FC_dropout_"):
            W = tf.get_variable("FC_weight",
                                initializer=tf.truncated_normal([LSTM_size, sum(num_filters)], mean=0, stddev=0.1))
            b = tf.get_variable("FC_bias",
                                initializer=tf.truncated_normal([sum(num_filters)], mean=0, stddev=0.1))

            self.FC_output1 = tf.nn.relu(tf.nn.xw_plus_b(self.states, W, b))
            self.FC_output_dropout = tf.nn.dropout(self.FC_output1, keep_prob=self.keep_prob)

        # scores and output
        with tf.name_scope("Scores_and_output"):
            W = tf.get_variable("output_weight",
                                initializer=tf.truncated_normal([sum(num_filters), num_classes], mean=0, stddev=0.1))
            b = tf.get_variable("output_bias",
                                initializer=tf.truncated_normal([num_classes], mean=0, stddev=0.1))

            self.scores = tf.nn.xw_plus_b(self.FC_output_dropout, W, b)
            self.output = tf.argmax(self.scores, axis=1)

        # loss/accuracy
        with tf.name_scope("loss_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label_y)
            self.loss = tf.reduce_mean(losses)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.label_y, axis=1)), 'float'))

    def cnn(self, input_x, filter_size, index, embedding_size, num_filters):
        with tf.name_scope("cnn_maxpool_%s" % index):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.get_variable("cnn_weight_%s" % index,
                                initializer=tf.truncated_normal(filter_shape, mean=0, stddev=0.01))
            b = tf.get_variable("cnn_bias_%s" % index,
                                initializer=tf.truncated_normal([num_filters], mean=0, stddev=0.01))

            # conv nets
            conv = tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv_net_%s" % index)

            # apply non-linearity
            h = tf.nn.bias_add(conv, b)

            # max-pooling
            seq_length = input_x.get_shape()[1].value
            ksize = [1, 2, 1, 1]
            pooling = tf.nn.max_pool(h, ksize=ksize, strides=[1, self.max_pooling_size, 1, 1], padding='VALID')
            return tf.nn.dropout(tf.reshape(pooling, shape=[-1, int((seq_length - filter_size + 1)/self.max_pooling_size), num_filters]),
                                 keep_prob=self.keep_prob)

class Training(data_tool, TextRNN):

    def __init__(self):
        self.epoch_size = 12
        self.batch_size = 128
        print("init data..")
        data_tool.__init__(self, train_path=train_path, test_path=test_path)

        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                print("init model..")
                TextRNN.__init__(self, sequence_len=self.max_length, embedding_size=200,
                                 num_classes=5, vocabulary_size=len(self.vocab_dict.keys()),
                                 LSTM_size=256,
                                 cnn_filer_size=[3, 4, 5], num_filters=[128, 128, 128],
                                 word_vec=None, max_pooling_size=2)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                self.saver = tf.train.Saver()

                optimizer = tf.train.AdamOptimizer(0.001)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step)

                # get real_length
                def real_length(batches):
                    return np.ceil([np.argmin(batch.tolist()+[0]) for batch in batches])

                # initialize variable
                sess.run(tf.global_variables_initializer())

                # generate batches
                batches_all = self.generate_batches(list(zip(self.train_x, self.train_y)), epoch_size=self.epoch_size,
                                                    batch_size=self.batch_size, shuffle=True)
                total_amount = (len(self.train_x) // self.batch_size + 1) * self.epoch_size
                for i, batch in enumerate(batches_all):
                    batch_x, batch_y = zip(*batch)
                    loss, _, accuracy, step = sess.run([self.loss, train_op, self.accuracy, global_step],
                                                       feed_dict={self.input_x: batch_x,
                                                                  self.label_y: batch_y,
                                                                  self.keep_prob: 0.5,
                                                                  self.real_seq_length: real_length(batch_x),
                                                                  self.padding: np.zeros(
                                                                      [len(batch_x), 1, self.embedding_size, 1])})

                    print("Currently at batch {}/{}".format(i, total_amount), "The loss is %f" % loss)
                    if i % 100 == 0:
                        print("current batch accuracy is:", accuracy)
                        self.saver.save(sess, "/tmp/model1.ckpt", global_step=i)

                # start testing training
                data_size = len(self.test_x)
                self.result = []
                for i in range(data_size // 500):
                    tmp = self.test_x[i * 500:(i + 1) * 500]
                    self.result.append(sess.run(self.output, feed_dict={self.input_x: tmp,
                                                                    self.keep_prob: 1.0,
                                                                    self.padding: np.zeros(
                                                                        [len(tmp), 1, self.embedding_size, 1]),
                                                                    self.real_seq_length: real_length(tmp)}
                                           ))
                tmp = self.test_x[(i+1)*500:]
                self.result.append(sess.run(self.output, feed_dict={self.input_x: self.test_x[(i + 1) * 500:],
                                                                self.keep_prob: 1.0,
                                                                self.padding: np.zeros(
                                                                    [len(tmp), 1, self.embedding_size, 1]),
                                                                self.real_seq_length: real_length(tmp)}))
                self.result_ = np.concatenate(self.result, axis=0)

                # self.save_data(self.result)

if __name__ == '__main__':
    train_ = Training()
