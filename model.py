import codecs
import tensorflow as tf
import numpy as np
import config as cfg

from random import shuffle
from gensim.models.keyedvectors import KeyedVectors
"""
Note: data should end with two new lines
"""
parameters = {}

encoding = cfg.encoding
type_of_emb = 'cnn'
learning_model = 'lstm'

def train_model():
    """
    training a model
    """
    train_const = True
    #  ===================================================
    # Those can be run once
    # model = KeyedVectors.load_word2vec_format('embeddings/word2vec.bin', binary=True)
    # model.save_word2vec_format('embeddings/word2vec.txt', binary=False)
    # save_all_words()
    # voc = load_vocabulary(cfg.vocabulary_location)
    # export_trimmed_vectors(voc)
    #
    # =============================================================

    word_vectors = get_trimmed_vectors(cfg.trimmed_embeddings_file)

    training_words, training_tags, _, _, _ = load_train_vocabulary(
        cfg.train_location)

    training_words_test, training_tags_test, _, _, _ = load_train_vocabulary(
        cfg.test_location)

    training_data_ids, training_tags_ids, chars_ids, words_to_ids, ids_to_words = build_data(
        training_words, training_tags)

    test_data_ids, test_tags_ids, chars_ids_test, test_id_to_words, _ = build_data(
        training_words_test, training_tags_test)

    placeholders = define_placeholders()

    if train_const:
        parameters = build_model(
            word_vectors, training_data_ids, words_to_ids, placeholders)
        sess, saver = initialize_session()
        train(sess, saver, training_data_ids, training_tags_ids, chars_ids,
              test_data_ids, test_tags_ids, chars_ids_test, parameters, placeholders)
    else:
        parameters = build_model(
            word_vectors, test_data_ids, test_id_to_words, placeholders)
        sess, saver = initialize_session()
        restore_session(cfg.model, saver, sess)

        evaluate(sess, test_data_ids, test_tags_ids,
                 chars_ids_test, placeholders, parameters)

    # run_epoch(sess, saver, training_data_ids, training_tags_ids, test_data_ids,
    #           test_tags_ids, test_id_to_words, parameters, placeholders)
    print('model is trained')


def train(sess, saver, training_data_ids, training_tags_ids, chars_ids, test_data_ids, test_tags_ids, chars_ids_test, parameters, placeholders):
    best_score = 0
    curr_nepoch_no_imprv = 0
    cfg.nepoch_no_imprv
    cfg.lr
    for epoch in range(cfg.epoch_range):

        print("Epoch Num:", epoch)
        score = run_epoch(sess, saver, training_data_ids, training_tags_ids, chars_ids,
                          test_data_ids, test_tags_ids, chars_ids_test, parameters, placeholders)
        # if epoch > 2 and epoch < 6:
        #     cfg.lr = 0.005
        cfg.lr = cfg.lr * cfg.lr_decay
        print('=============================')
        print('current score', score)
        print('best score', best_score)
        if score >= best_score:
            curr_nepoch_no_imprv = 0
            save_session(sess, saver, cfg.model)
            best_score = score
        else:
            print('no improvements, no improvement: ', curr_nepoch_no_imprv)
            curr_nepoch_no_imprv += 1
            if curr_nepoch_no_imprv >= cfg.nepoch_no_imprv:
                print('no improvements')
                break


def run_epoch(sess, saver, training_data_ids, training_tags_ids, chars_ids, test_data_ids, test_tags_id, chars_ids_test, parameters, placeholders):

    batches_training, batches_tags, batches_chars_ids = create_batches(
        training_data_ids, chars_ids, training_tags_ids, cfg.batches_size)

    for i, (batch_training, batch_tags, batch_chars_ids) in enumerate(zip(batches_training, batches_tags, batches_chars_ids)):
        # print('batch', i)

        fd, _ = get_feed(batch_training, batch_chars_ids,
                         placeholders, batch_tags, cfg.lr, cfg.dropout)
        # break
        _, train_loss = sess.run(
            [parameters['train_op'], parameters['loss']], feed_dict=fd)

    print('train_loss', train_loss)

    f1 = evaluate(sess, test_data_ids, test_tags_id,
                  chars_ids_test, placeholders, parameters)
    return f1


def create_batches(training_data, chars_ids, training_tags, n):
    a = [training_data[i:i + n] for i in range(0, len(training_data), n)]
    b = [training_tags[i:i + n] for i in range(0, len(training_tags), n)]
    c = [chars_ids[i:i + n] for i in range(0, len(chars_ids), n)]

    return a, b, c


def create_batches_test(training_data, chars_ids, training_tags, n):

    x_batch, y_batch, z_batch = [], [], []
    for (x, y, z) in zip(training_data, training_tags, chars_ids):
        if len(x_batch) == n:
            yield x_batch, y_batch, z_batch
            x_batch, y_batch, z_batch = [], [], []

        x_batch += [x]
        y_batch += [y]
        z_batch += [z]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch


def get_feed(training_data, batch_chars_ids, placeholders, labels=None, lr=None, dropout=None):

    word_ids_, sequence_lengths_ = pad_sequence(training_data)
    char_ids_, word_lengths_ = pad_sequence_chars(batch_chars_ids)

    feed = {
        placeholders['word_ids']: word_ids_,
        placeholders['sequence_lengths']: sequence_lengths_,
        placeholders['char_ids']: char_ids_,
        placeholders['word_lengths']: word_lengths_
    }
   
    if labels is not None:

        labels_, _ = pad_sequence(labels)
        feed[placeholders['labels']] = labels_

    if lr is not None:
        feed[placeholders['lr']] = cfg.lr

    if dropout is not None:
        feed[placeholders['dropout']] = cfg.dropout

    # print (feed)
    return feed, sequence_lengths_


def load_train_vocabulary(path):

    sentences = []
    sentence = []
    vocabulary = set()
    vocabulary_for_chars = set()
    training_words = []
    training_tags = []
    tags = []
    voc_tags = set()
    for line in codecs.open(path, 'r', encoding):

        if line == "\n" or line == "\r\n":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line)

    for s in sentences:

        sent_words = []
        sent_tags = []
        for word in s:

            w = word.split(' ')

            voc_tags.add(w[-1].rstrip('\r\n'))

            sent_words.append(w[0])
            sent_tags.append(w[-1].rstrip('\r\n'))
        training_words.append(sent_words)
        training_tags.append(sent_tags)

    for ws in training_words:
        for wws in ws:
            wws = wws.strip()
            if wws.isdigit():
                wws = "$NUM$"

            else:
                vocabulary.add(wws.lower())
                vocabulary_for_chars.add(wws)

    return training_words, training_tags, vocabulary, voc_tags, vocabulary_for_chars


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words in dataset:
        for word in words:
            if word != ' ':
                vocab_char.update(word)

    return vocab_char




def load_fasttext_vocab(pretrained_vectors_location):
    print (' loading fast text')
    vocab = set()
    with open(pretrained_vectors_location, encoding="utf-8") as f:
        for line in f:

            word = line.strip().split(' ')[0]

            vocab.add(word.lower())

    return vocab


def save_vocabulary(vocabulary, location):
    f = codecs.open(location, 'w', encoding)
    for i, word in enumerate(vocabulary):

        f.write(word)
        f.write("\n")


def load_vocabulary(vocabulary_location):
    d = dict()
    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0].encode(encoding)

        d[line] = i

        i += 1

    return d


def load_vocabulary_to_id(vocabulary_location):
    d = dict()
    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0].encode(encoding)

        d[i] = line

        i += 1

    return d


def load_vocabulary_tags(vocabulary_location):
    d = dict()

    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0]

        d[line] = i

        i += 1

    return d


def load_vocabulary_tags_keys(vocabulary_location):
    d = dict()
    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0]

        d[i] = line

        i += 1

    return d


def export_trimmed_vectors(vocab):

    embeddings = np.zeros([len(vocab), cfg.dim])
    with open(cfg.pretrained_vectors_location, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]

            embedding = [float(x) for x in line[1:]]
            if vocab.get(word.encode(encoding)) != None:
                word_idx = vocab[word.encode(encoding)]

                embeddings[word_idx] = np.asarray(embedding)
   
    np.savez_compressed(cfg.trimmed_embeddings_file, embeddings=embeddings)


def get_trimmed_vectors(trimmed_embeddings_file):
    with np.load(trimmed_embeddings_file) as data:

        return data["embeddings"]


def save_all_words():

    _, _, training_words, training_tags, voc_for_chars = load_train_vocabulary(
        cfg.train_location)
    _, _, training_words_test, training_tags_test, _ = load_train_vocabulary(
        cfg.test_location)
    _, _, training_words_dev, training_tags_dev, _ = load_train_vocabulary(
        cfg.dev_location)

    fast_words = load_fasttext_vocab(cfg.pretrained_vectors_location)
    # print(training_words)
    # comment fast words if nothing else is working...

    vocab_words_ = training_words | training_words_test | training_words_dev

    vocab_words = vocab_words_ & fast_words
    vocab_tags = training_tags | training_tags_test | training_tags_dev
    vocab_words.add("$UNK$")

    vocab_chars = get_char_vocab(voc_for_chars)
    vocab_chars.add("$UNK$")
    save_vocabulary(vocab_words, cfg.vocabulary_location)
    save_vocabulary(vocab_tags, cfg.vocabulary_tags_location)
    save_vocabulary(vocab_chars, cfg.vocabulary_chars_location)


def build_model(wordVectors, training_data_ids, words_to_ids, placeholders):
    """ Get the training data - sentence by sentence and run it through parse_sentence_to_vectors function.
    This way we will have a sequential vectors for every word in every sentence a long with their correct tags.

    Having this information we can load it into a fully connected neural network. The result: for every word -
    a vector with scores for each class.

    Picking the highest class score for every word won't give us the best result as the sequence might be wrong.
    """
    inputs = []
    embedding_sentences = []
    outputs = []
    training_tags = []
    vocabulary = []
    count_used = 0
    count_all = 0
    # wordVectors = load_pretrained_vec(cfg.pretrained_vectors_location, training_data_ids, words_to_ids)

    # Call all batches in a loop
    # To be added
    # -------------------------------------------

    word_embeddings = add_embeddings(wordVectors, placeholders)

    parameters['logits'] = add_logits_op(word_embeddings, placeholders)
    parameters['labels_pred'] = add_pred_op(parameters['logits'])
    #  Operations, remove them from parameters and put them in operations
    parameters['loss'], parameters['trans_params'] = add_loss_op(
        parameters['logits'], placeholders)
    parameters['train_op'] = add_train_op(
        cfg._lr_m, cfg.lr, parameters['loss'], placeholders)

    print('model is finally built')
    return parameters


def define_placeholders():
    word_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="word_ids")
    labels_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                        name="labels")
    sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=[None],
                                                  name="sequence_lengths")
    dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[],
                                         name="dropout")
    lr_learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="lr")
    word_lengths_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                              name="word_lengths")
    char_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None, None],
                                          name="char_ids")
    placeholders = {
        'labels': labels_placeholder,
        'word_ids': word_ids_placeholder,
        'word_lengths': word_lengths_placeholder,
        'sequence_lengths': sequence_lengths_placeholder,
        'char_ids': char_ids_placeholder,
        'dropout': dropout_placeholder,
        'lr': lr_learning_rate
    }
    return placeholders


def add_embeddings(wordVectors, placeholders):
    nchars = len(load_vocabulary(cfg.vocabulary_chars_location))
    nwords = len(load_vocabulary(cfg.vocabulary_location))
    # with tf.variable_scope("words"):
    #     _word_embeddings = tf.get_variable(
    #         name="_word_embeddings",
    #         dtype=tf.float32,
    #         shape=[nwords, dim])

    with tf.variable_scope("words"):
        _word_embeddings = tf.Variable(
            wordVectors,
            name="_word_embeddings",
            dtype=tf.float32,
            trainable=False)
        word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                 placeholders['word_ids'], name="word_embeddings")
    if type_of_emb == 'lstm':
        with tf.variable_scope("chars"):

            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[nchars, cfg.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                    placeholders['char_ids'], name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                        shape=[s[0] * s[1], s[-2], cfg.dim_char])
            word_lengths = tf.reshape(
                placeholders['word_lengths'], shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output,
                                shape=[s[0], s[1], 2 * cfg.hidden_size_char])
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        word_embeddings_ = tf.nn.dropout(word_embeddings, placeholders['dropout'])

        return word_embeddings_

    if type_of_emb == 'cnn':
       
        token_lengths = tf.placeholder(tf.int32, [None, None], name="tok_lengths")
        input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        with tf.variable_scope("char-forward", reuse=False):
            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[nchars, cfg.dim_char])
            char_embeddings_lookup = tf.nn.embedding_lookup(_char_embeddings, placeholders['char_ids'])
            s = tf.shape(char_embeddings_lookup)
            max_token_len =  tf.reshape(
                    placeholders['word_lengths'], shape=[s[0] * s[1]])
            # max_token_len = placeholders['word_lengths']
            print(char_embeddings_lookup.get_shape())

            char_embeddings_flat = tf.reshape(char_embeddings_lookup, 
            tf.stack([cfg.batches_size*placeholders['sequence_lengths'], max_token_len, cfg.dim]))
            print(char_embeddings_flat.get_shape())
            tok_lens_flat = tf.reshape(token_lengths, [cfg.batches_size*placeholders['sequence_lengths']])
            print(tok_lens_flat.get_shape())

            input_feats_expanded = tf.expand_dims(char_embeddings_flat, 1)
            input_feats_expanded_drop = tf.nn.dropout(input_feats_expanded, input_dropout_keep_prob)


            with tf.name_scope("char-cnn"):
                filter_shape = [1, 0, 100, 100]

                w = tf.get_variable("conv0_w", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                # w = tf.get_variable("conv0_w", shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("conv0_b", initializer=tf.constant(0.01, shape=[100]))
                conv0 = tf.nn.conv2d(input_feats_expanded_drop, w, strides=[1, 1, 1, 1], padding="SAME", name="conv0")
                print("conv0", conv0.get_shape())
                h_squeeze = tf.squeeze(conv0, [1])
                print("squeeze", h_squeeze.get_shape())
                hidden_outputs = tf.reduce_max(h_squeeze, 1)
                print("max", hidden_outputs.get_shape())
                hidden_outputs_unflat = tf.reshape(hidden_outputs, tf.stack([cfg.batches_size, placeholders['sequence_lengths'], 100]))

        return hidden_outputs_unflat

    if type_of_emb == 'att':
        with tf.variable_scope("chars"):

            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[nchars, cfg.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                    placeholders['char_ids'], name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                        shape=[s[0] * s[1], s[-2], cfg.dim_char])
            word_lengths = tf.reshape(
                placeholders['word_lengths'], shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output,
                                shape=[s[0], s[1], 2 * cfg.hidden_size_char])

            # Output is the chars representation
            # ===========================================================
            with tf.variable_scope("attention"):
                print("Output vector size",output.shape)
                #         W = tf.get_variable("W", dtype=tf.float32,shape=[2 * cfg.hidden_size_lstm, cfg.ntags])
                # initializer=tf.zeros_initializer()
                weights_words = tf.Variable(tf.random_normal([ cfg.batches_size,   cfg.dim , cfg.hidden_size_lstm] ))
                weights_chars = tf.Variable(tf.random_normal([cfg.batches_size, 2 * cfg.dim_char, 2 * cfg.hidden_size_char ]))
                weights_all = tf.Variable(tf.random_normal([cfg.batches_size,  2* cfg.dim_char + cfg.dim,  2 * cfg.hidden_size_char + cfg.hidden_size_lstm]))

                # weights_words = tf.Variable(tf.random_normal([ cfg.batches_size,  cfg.hidden_size_lstm,  cfg.dim ]))
                # weights_chars = tf.Variable(tf.random_normal([cfg.batches_size,  2 * cfg.hidden_size_char,2 * cfg.dim_char]))
                # weights_all = tf.Variable(tf.random_normal([cfg.batches_size,   2 * cfg.hidden_size_char + cfg.hidden_size_lstm, 2 * cfg.dim_char + cfg.dim]))
            
                # word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                h1    = tf.nn.sigmoid(tf.matmul(word_embeddings, weights_words))  # The \sigma function

                h2    = tf.nn.sigmoid(tf.matmul(output, weights_chars))

                s = tf.shape(h2)
                # h2 = tf.reshape(char_embeddings,
                #                         shape=[s[0] * s[1], s[-2], cfg.dim])

                # h2 = tf.reshape(h2,
                #                      shape=[h2.shape[0], h2.shape[1], h1.shape[2]])

                res = tf.tanh(tf.concat([h1, h2], axis = -1))

                print('resssss shape', res.shape)
                h3 = tf.nn.sigmoid(tf.matmul(res, weights_all))
                # a = tf.matmul(res, weights_all)
                # print('a:', a)
                # h3 = tf.nn.tanh(tf.matmul(res, weights_all))
                # # h3 = tf.reshape(h3, shape = [-1, -1,h3.shape[2]])
                # print('shape h3', h3.shape)
                # Current issue is with dropout of h3!!!!!!!!!!!!!!!
                print('shape of h2:', h2)
                # print('shape of h3', h3)
        word_embeddings_ = tf.nn.dropout(h3, placeholders['dropout'])
        
        return word_embeddings_

def add_logits_op(word_embeddings, placeholders):
    """Defines logits

    For each word in each sentence of the batch, it corresponds to a vector
    of scores, of dimension equal to the number of tags.
    """
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('how word embeddings should look like:', word_embeddings)

    if learning_model == "lstm":
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, word_embeddings,
                sequence_length=placeholders['sequence_lengths'], dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            # print('lstm output shape', output.shape)
            output = tf.nn.dropout(output, placeholders['dropout'])

        with tf.variable_scope("proj"):

            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * cfg.hidden_size_lstm, cfg.ntags])

            b = tf.get_variable("b", shape=[cfg.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * cfg.hidden_size_lstm])
            print('=============')
            # print(output.shape)
            pred = tf.matmul(output, W) + b
            logits = tf.reshape(pred, [-1, nsteps, cfg.ntags])
        return logits

    if learning_model == 'cnn':
        print ('a')


def add_pred_op(logits):
    """

    This op is defined only in the case where we don't use a CRF since in
    that case we can make the prediction "in the graph" (thanks to tf
    functions in other words). With theCRF, as the inference is coded
    in python and not in pure tensroflow, we have to make the prediciton
    outside the graph.
    """
    labels_pred = tf.cast(tf.argmax(logits, axis=-1),
                          tf.int32)
    return labels_pred


def add_loss_op(logits, placeholders):
    """Defines the loss"""
    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        logits, placeholders['labels'], placeholders['sequence_lengths'])
    trans_params = trans_params  # need to evaluate it for decoding
    loss = tf.reduce_mean(-log_likelihood)

    return loss, trans_params


def add_train_op(_lr_m, lr, loss, placeholders):
    with tf.variable_scope("train_step"):
        if _lr_m == 'adam':
            optimizer = tf.train.AdamOptimizer(placeholders['lr'])
        elif _lr_m == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(placeholders['lr'])
        elif _lr_m == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(placeholders['lr'])
        elif _lr_m == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=placeholders['lr'], momentum=0.9, use_nesterov=True)
        elif _lr_m == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(placeholders['lr'])
        else:
            raise NotImplementedError("Unknown method {}".format(_lr_m))
        if cfg.clip > 0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm = tf.clip_by_global_norm(grads, cfg.clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)

    return train_op


def initialize_session():
    """Defines sess and initialize the variables"""
    # logger.info("Initializing tf session")
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    return sess, saver


def pad_sequence(training_data):
    sequence_padded = []
    sequence_length = []

    max_length = max(map(lambda x: len(x), training_data))
    # max_length = 20
    for data in training_data:
        data = list(data)
        data_ = data[:max_length] + [0] * max(max_length - len(data), 0)
        sequence_padded += [data_]
        sequence_length += [min(len(data), max_length)]
    # print(sequence_length)
    return sequence_padded, sequence_length


def pad_sequence_chars(sequences):

    sequence_padded = []
    sequence_length = []

    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, 0, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded,
                                        [0] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0,
                                        max_length_sentence)
    return sequence_padded, sequence_length


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def convert_words_to_ids(location):
    words_to_ids = {}
    ids_to_words = {}
    i = 0
    for line in codecs.open(location, 'r', encoding):
        line = line.split('\n')[0].encode(encoding)

        words_to_ids[line] = i
        ids_to_words[i] = line
        i += 1

    return words_to_ids, ids_to_words


def convert_tags_to_ids(location):
    words_to_ids = {}
    ids_to_words = {}
    i = 0
    for line in codecs.open(location, 'r', encoding):
        line = line.split('\n')[0]

        words_to_ids[line] = i
        ids_to_words[i] = line
        i += 1

    return words_to_ids, ids_to_words


def randomize_data():
    lines = []
    f1 = codecs.open("./dataset/bg/train.txt", "rb", encoding)
    f2 = codecs.open("dataset/bg/testa.txt", "rb", encoding)
    f3 = codecs.open("dataset/bg/testb.txt", "rb", encoding)
    sentences = []
    sentence = []
    for line in f1:
        if line == "\r\n":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line.strip())

    for line2 in f2:
        if line2 == "\r\n":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line2.strip())
    for line3 in f3:
        if line3 == "\r\n":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line3.strip())

    shuffle(sentences)
    f = codecs.open("all_without_bulletin.txt", "w", "utf-8")

    for sen in sentences:
        for s in sen:
            f.write(s)
            f.write("\n")
        f.write('\n')


def yield_data(location):
    niter = 0
    with open(location, encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    niter += 1

                    yield words, tags
                    words, tags = [], []
            else:
                ls = line.split(' ')
                word, tag = ls[0], ls[-1]

                words += [word]
                tags += [tag]


def build_data_test(location):

    training_data_ids = []
    training_tags_ids = []
    ids_data = []
    ids_tags = []
    words_to_ids, ids_to_words = convert_words_to_ids(cfg.vocabulary_location)
    tags_to_ids, ids_to_tags = convert_words_to_ids(
        cfg.vocabulary_tags_location)
    chars_to_ids, _ = convert_words_to_ids(cfg.vocabulary_chars_location)

    chars_in_word = []
    chars_words = []
    list_chars = []
    for sentence_words, sentence_tags in yield_data(location):
        for word, tag in zip(sentence_words, sentence_tags):

            word = word.lower()
            if words_to_ids.get(word) != None:
                ids_data.append(words_to_ids[word])

                for w in list(word.decode(encoding)):
                    if chars_to_ids.get(w.encode(encoding)) != None:
                        chars_in_word.append(chars_to_ids[w.encode(encoding)])
                chars_words.append(chars_in_word)
            else:
                f = codecs.open("./results/test.txt", 'w', encoding)
                f.write(word)
                f.write('\n')
            if tags_to_ids.get(tag) != None:
                ids_tags.append(tags_to_ids.get(tag))
        training_data_ids.append(ids_data)
        ids_data = []
        training_tags_ids.append(ids_tags)
        ids_tags = []
        list_chars.append(chars_words)

    return training_data_ids, training_tags_ids, list_chars, words_to_ids, ids_to_words


def build_data(training_data, training_tags):

    training_data_ids = []
    training_tags_ids = []
    ids_data = []
    ids_tags = []
    words_to_ids, ids_to_words = convert_words_to_ids(cfg.vocabulary_location)
    tags_to_ids, ids_to_tags = convert_tags_to_ids(
        cfg.vocabulary_tags_location)

    chars_to_ids, _ = convert_words_to_ids(cfg.vocabulary_chars_location)

    chars_in_word = []
    chars_words = []
    list_chars = []
    unk = "$UNK$".encode(encoding)
    for sentence_words, sentence_tags in zip(training_data, training_tags):

        for word, tag in zip(sentence_words, sentence_tags):

            # word = word.lower()

            # word = word.decode(encoding).replace(" ", "").encode(encoding)
            if words_to_ids.get(word.lower().encode(encoding)) != None:
                ids_data.append(words_to_ids[word.lower().encode(encoding)])
            else:

                ids_data.append(words_to_ids[unk])
            word = word.encode(encoding)
            for w in list(word.decode(encoding)):
                if chars_to_ids.get(w.encode(encoding)) != None:
                    chars_in_word.append(chars_to_ids[w.encode(encoding)])
                else:
                    chars_in_word.append(chars_to_ids[unk])

            chars_words.append(chars_in_word)
            chars_in_word = []
            if tags_to_ids.get(tag) != None:
                ids_tags.append(tags_to_ids[tag])
        training_data_ids.append(ids_data)

        ids_data = []
        training_tags_ids.append(ids_tags)
        ids_tags = []
        list_chars.append(chars_words)
        chars_words = []

    return training_data_ids, training_tags_ids, list_chars, words_to_ids, ids_to_words


def save_session(sess, saver, location):
    saver.save(sess, location)


def restore_session(location, saver, sess):
    print('Loading latest trained model')
    saver.restore(sess, location)


def predict_batch(sess, data, chars_ids_test, placeholders, parameters):
    # sess, saver = initialize_session()
    # restore_session('./results/model', saver, sess)

    fd, sequence_lengths = get_feed(
        data, chars_ids_test, placeholders, dropout=1.0)

    # get tag scores and transition params of CRF
    viterbi_sequences = []
    logits, trans_params = sess.run(
        [parameters['logits'], parameters['trans_params']], feed_dict=fd)

    # iterate over the sentences because no batching in vitervi_decode
    for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length]  # keep only the valid steps

        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
            logit, trans_params)
        viterbi_sequences += [viterbi_seq]

    return viterbi_sequences, sequence_lengths

    
    # labels_pred = sess.run(parameters['labels_pred'], feed_dict=fd)

    # return labels_pred, sequence_lengths


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    # tags = {}
    # for i, tag in enumerate(tags_arr):
    #     # tag = tag.decode(encoding)
    #     tags[tag] = i

    default = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def load_vocabulary_list(fif):
    tags = []
    ff = codecs.open(fif, "rb", encoding)
    for line in ff:
        tags.append(line.strip())
    return tags


def evaluate(sess, test_data_ids, test_tags_id, chars_ids_test, placeholders, parameters):
    # batches_training_test, batches_tags_test = create_batches(test_data_ids, test_tags_id, n_test)
    f = codecs.open(cfg.result_location, 'w', encoding)

    accs = []
    tags = load_vocabulary_tags(cfg.vocabulary_tags_location)
    tags_ = load_vocabulary_tags_keys(cfg.vocabulary_tags_location)
    words = load_vocabulary_to_id(cfg.vocabulary_location)
    correct_preds, total_correct, total_preds = 0., 0., 0.

    correct = 0
    overall = 0
    equal = 0
    count = 0
    O, b_loc, i_loc, b_org, i_org, b_pers, i_pers, b_oth, i_oth = 0, 0, 0, 0, 0, 0, 0, 0, 0
    O_all, b_loc_all, i_loc_all, b_org_all, i_org_all, b_pers_all, i_pers_all, b_oth_all, i_oth_all = 0, 0, 0, 0, 0, 0, 0, 0, 0

    batches_training, batches_tags, chars_ids_test_batch = create_batches(
        test_data_ids, chars_ids_test, test_tags_id, cfg.batches_size)

    for i, (batch_words, batch_tags, batch_chars) in enumerate(zip(batches_training, batches_tags, chars_ids_test_batch)):

        labels_pred, sequence_lengths = predict_batch(
            sess, batch_words, batch_chars, placeholders, parameters)

        for lab, lab_pred, length, sentence in zip(batch_tags, labels_pred,
                                                   sequence_lengths, batch_words):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            for a, b, s in zip(lab, lab_pred, sentence):
                f.write(words.get(s).decode(encoding))
                f.write(' ')
                f.write(tags_[a])
                f.write(' ')
                f.write(tags_[b])
                f.write("\n")
            f.write("\n")
            lab_chunks = set(get_chunks(lab, tags))

            lab_pred_chunks = set(get_chunks(lab_pred, tags))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
    # print("correct predictions", correct_preds)
    # print("total predictions", total_preds)
    # print("total correct", total_correct)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)
    # print("precision", p)
    # print("recall", r)

    print("acc", 100 * acc, "f1", f1 * 100)
    return f1 * 100


def run_evaluate(location):
    print('a')
    # labels_pred, sequence_lengths = predict_batch(
    #         sess, batch_words, batch_chars, placeholders, parameters)

    # for lab, lab_pred, length, sentence in zip(batch_tags, labels_pred,
    #                                             sequence_lengths, batch_words):

    #     lab = lab[:length]
    #     lab_pred = lab_pred[:length]

    # training_data_ids, training_tags_ids, words_to_ids, ids_to_words = build_data(
    #     location)
    # test_data_ids, test_tags_ids, _, _ = build_data(test_location)
    # placeholders = define_placeholders()
    # parameters = build_model(training_data_ids, words_to_ids, placeholders)
    # sess, saver = initialize_session()
    # n = int(len(training_data_ids) / num_batches)
    # n_test = int(len(test_data_ids) / num_batches)

    # batches_training, batches_tags = create_batches(
    #     training_data_ids, training_tags_ids, n)

    # evaluate(test_data_ids, test_tags_id, placeholders, parameters)


def main():
    # randomize_data()
    train_model()
    # run_evaluate('./dataset/bg/testb.txt')
    # get_fasttext_vocab(pretrained_vectors_location)
    # evaluate('./dataset/bg/testa.txt')


if __name__ == '__main__':
    main()
