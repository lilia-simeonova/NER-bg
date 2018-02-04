import codecs
import tensorflow as tf
import numpy as np
"""
training a model 
"""
encoding = 'utf-8'
pretrained_vectors_location = './embeddings/wiki.bg.vec'
vocabulary_location = './vocabulary/words'
model_location = './results/model'
maxSeqLength = 50
_lr_m = 'sgd' 
dropout = 0.5
hidden_size_lstm = 300
dim = 300
ntags = 9
lr = 0.001
lr_decay = 0.9
num_batches = 2
parameters = {}
train_location = './dataset/bg/random1.txt'
test_location = './dataset/bg/random2.txt'
trimmed_embeddings_file = 'embeddings/trimmed.npz'
batches_size = 20
nepoch_no_imprv = 5


def train_model():
    """
    training a model 
    """
    training_words, training_tags, _, _ = load_train_vocabulary(train_location)
    training_words_test, training_tags_test,_,_ = load_train_vocabulary(test_location)
    training_data_ids, training_tags_ids, words_to_ids, ids_to_words = build_train_data(training_words, training_tags )
    #  ===================================================
    # Those can be run once
    # save_all_words()
    # voc = load_vocabulary(vocabulary_location)
    # export_trimmed_vectors(voc)
    word_vectors = get_trimmed_vectors(trimmed_embeddings_file)

    test_data_ids, test_tags_ids, test_id_to_words, _ = build_train_data(training_words_test, training_tags_test)
    placeholders = define_placeholders()
    parameters = build_model(word_vectors, training_data_ids, words_to_ids, placeholders)
    sess, saver = initialize_session()
    train(sess, saver, training_data_ids, training_tags_ids, test_data_ids, test_tags_ids, test_id_to_words, parameters, placeholders)
    run_epoch(sess, saver, training_data_ids, training_tags_ids, test_data_ids, test_tags_ids, test_id_to_words, parameters, placeholders)
    print('model is trained')

def train(sess, saver, training_data_ids, training_tags_ids, test_data_ids, test_tags_ids, test_id_to_words, parameters, placeholders):
    best_score = 0
    curr_nepoch_no_imprv = 0
    global nepoch_no_imprv
    global lr
    for epoch in range(10):
        score = run_epoch(sess, saver, training_data_ids, training_tags_ids, test_data_ids, test_tags_ids, test_id_to_words, parameters, placeholders)
        lr = lr * lr_decay
        print('new lr', lr)
        if score >= best_score:
            curr_nepoch_no_imprv = 0
            save_session(sess, saver, model_location)
            best_score = score
        else:
            nepoch_no_imprv += 1
            if curr_nepoch_no_imprv >= nepoch_no_imprv:
                print('no improvements')
                break

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
    placeholders = {
         'labels': labels_placeholder,
         'word_ids': word_ids_placeholder,
         'sequence_lengths': sequence_lengths_placeholder,
         'dropout': dropout_placeholder,
         'lr': lr_learning_rate
    }
    return placeholders

def run_epoch(sess, saver, training_data_ids, training_tags_ids, test_data_ids, test_tags_id, test_id_to_words, parameters, placeholders):

    batches_training, batches_tags = create_batches(training_data_ids, training_tags_ids, batches_size)
    
    for i, (batch_training, batch_tags) in enumerate(zip(batches_training, batches_tags)):
        print('batch', i)
        for tr, ts in zip(batch_training, batch_tags):
            if len(tr) != len(ts):
                print('difff')
                print (tr)
                print(ts)
        # if i == 47:
        # print('batch training shape:', batch_training.shape)
        # print('batch tags shape', batch_tags.shape)
        fd, _ = get_feed(batch_training, placeholders, batch_tags, lr, dropout)
        _, train_loss = sess.run( [parameters['train_op'], parameters['loss']], feed_dict=fd)
        print('train_loss', train_loss)


    f1 = evaluate(sess, test_data_ids, test_tags_id, test_id_to_words, placeholders, parameters)
    return f1


def create_batches(training_data, training_tags, n):
    a = [training_data[i:i+n] for i in range(0, len(training_data), n)]
    b = [training_tags[i:i+n] for i in range(0, len(training_tags), n)]

    return a, b

def get_feed(training_data, placeholders, labels=None, lr=None, dropout=None):


    word_ids_, sequence_lengths_ = pad_sequence(training_data)

    feed = {
        placeholders['word_ids']: word_ids_,
        placeholders['sequence_lengths']: sequence_lengths_
    }
    if labels is not None:

        labels_, _ = pad_sequence(labels)
        feed[placeholders['labels']] = labels_


    if lr is not None:
        feed[placeholders['lr']] = lr

    if dropout is not None:
        feed[placeholders['dropout']] = dropout

    
    return feed, sequence_lengths_

def load_train_vocabulary(path):
    sentences = []
    sentence = []
    vocabulary = set()
    training_words = []
    training_tags = []
    tags = []
    voc_tags = set()
    for line in codecs.open(path, 'r', encoding):
        line = line.encode(encoding)
        if line.decode(encoding) == "\r\n":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line)

    for s in sentences:

        sent_words = []
        sent_tags = []
        for word in s:
            word = word.decode(encoding).split(u'\t')
            word_ = word[0].lower().encode(encoding)
            tag = word[2].rstrip('\r\n').encode(encoding)
            
            
            vocabulary.add(word[0].lower())
            
            voc_tags.add(word[2].rstrip('\r\n'))
            sent_words.append(word_)
            sent_tags.append(tag)
        training_words.append(sent_words)
        training_tags.append(sent_tags)
             
    # print(vocabulary)

    return training_words, training_tags, vocabulary, voc_tags


def load_fasttext_vocab(pretrained_vectors_location):
    print("Building vocab...")
    vocab = set()
    pretrained_embeddings = codecs.open(
        pretrained_vectors_location, 'rb', encoding)
   
    for line in pretrained_embeddings:
        if line:

            word = line.lstrip("„").lower().rstrip()
            if word:
                word_ = word.split()[0]

            vocab.add(word_)
    print("- done. {} tokens".format(len(vocab)))

    return vocab

def save_all_words():

    _, _, training_words, training_tags = load_train_vocabulary(train_location)
    _, _, training_words_test, training_tags_test = load_train_vocabulary(test_location)


    # fast_words = load_fasttext_vocab(pretrained_vectors_location)
    # print(training_words)

    vocab_words_ = training_words | training_words_test
    vocab_words = vocab_words_ 
    vocab_tags = training_tags | training_tags_test

    save_vocabulary(vocab_words, './vocabulary/words')
    save_vocabulary(vocab_tags, './vocabulary/tags')



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
    # wordVectors = load_pretrained_vec(pretrained_vectors_location, training_data_ids, words_to_ids)
       
    # Call all batches in a loop
    # To be added 
    # -------------------------------------------

    word_embeddings = add_embeddings(wordVectors, placeholders)

    parameters['logits'] = add_logits_op(word_embeddings, placeholders)
    # parameters['labels_pred'] = add_pred_op(parameters['logits'])
    #  Operations, remove them from parameters and put them in operations
    parameters['loss'], parameters['trans_params'] = add_loss_op(parameters['logits'], placeholders)
    parameters['train_op'] = add_train_op(_lr_m, lr, parameters['loss'], placeholders)


    print('model is finally built')
    return parameters

def add_embeddings(wordVectors, placeholders):

    with tf.variable_scope("words"):
        _word_embeddings = tf.Variable(
                            wordVectors,
                            name="_word_embeddings",
                            dtype=tf.float32,
                            trainable=False)
        word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                        placeholders['word_ids'], name="word_embeddings")

        word_embeddings_ =  tf.nn.dropout(word_embeddings, placeholders['dropout'])

    #  with tf.variable_scope("chars"):

            # _char_embeddings = tf.get_variable(
            #         name="_char_embeddings",
            #         dtype=tf.float32,
            #         shape=[self.config.nchars, 100])
            # char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
            #         self.char_ids, name="char_embeddings")

            # # put the time dimension on axis=1
            # s = tf.shape(char_embeddings)
            # char_embeddings = tf.reshape(char_embeddings,
            #         shape=[s[0]*s[1], s[-2], self.config.dim_char])
            # word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

            # # bi lstm on chars
            # cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
            #         state_is_tuple=True)
            # cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
            #         state_is_tuple=True)
            # _output = tf.nn.bidirectional_dynamic_rnn(
            #         cell_fw, cell_bw, char_embeddings,
            #         sequence_length=word_lengths, dtype=tf.float32)

            # # read and concat output
            # _, ((_, output_fw), (_, output_bw)) = _output
            # output = tf.concat([output_fw, output_bw], axis=-1)

            # # shape = (batch size, max sentence length, char hidden size)
            # output = tf.reshape(output,
            #         shape=[s[0], s[1], 2*self.config.hidden_size_char])
            # word_embeddings = tf.concat([word_embeddings, output], axis=-1)
    print(word_embeddings_)
    return word_embeddings_

def add_logits_op(word_embeddings, placeholders):
    """Defines logits

    For each word in each sentence of the batch, it corresponds to a vector
    of scores, of dimension equal to the number of tags.
    """
    with tf.variable_scope("bi-lstm"):
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, word_embeddings,
                sequence_length=placeholders['sequence_lengths'], dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, placeholders['dropout'])

    with tf.variable_scope("proj"):
        W = tf.get_variable("W", dtype=tf.float32,
                shape=[2*hidden_size_lstm, ntags])

        b = tf.get_variable("b", shape=[ntags],
                dtype=tf.float32, initializer=tf.zeros_initializer())

        nsteps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2*hidden_size_lstm])
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, nsteps, ntags])
    return logits


def add_pred_op(logits):
    """Defines self.labels_pred

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
    print('Losssss operation run')
    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            logits, placeholders['labels'], placeholders['sequence_lengths'])
    trans_params = trans_params # need to evaluate it for decoding
    loss = tf.reduce_mean(-log_likelihood)
    
    return loss, trans_params

def add_train_op(_lr_m, lr, loss, placeholders):
    print('Train operation run')
    with tf.variable_scope("train_step"):
        if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(placeholders['lr'])
        elif _lr_m == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(placeholders['lr'])
        elif _lr_m == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(placeholders['lr'])
        elif _lr_m == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(placeholders['lr'])
        else:
            raise NotImplementedError("Unknown method {}".format(_lr_m))

        train_op = optimizer.minimize(loss)

    return train_op

def initialize_session():
    """Defines sess and initialize the variables"""
    # logger.info("Initializing tf session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    return sess, saver

def save_vocabulary(vocabulary, location):
    f = codecs.open(location, 'w', encoding)
    for i, word in enumerate(vocabulary):
        if i != len(vocabulary) - 1:
            # print(word.encode(encoding))
            f.write("{}\n".format(word))
        else:
            f.write(word)
    print('voc saved')

def load_vocabulary(vocabulary_location):
    d = dict()
    ff = codecs.open(vocabulary_location, "rb", encoding)
    for line in ff:
        for idx, word in enumerate(ff):
            word = word.strip()
            d[word] = idx
    return d

def export_trimmed_vectors(vocab):
    embeddings = np.zeros([len(vocab), dim])
    with open(pretrained_vectors_location, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_embeddings_file, embeddings=embeddings)

def get_trimmed_vectors(trimmed_embeddings_file):
    with np.load(trimmed_embeddings_file) as data:
            return data["embeddings"]

def change_load_pretrained_vec(pretrained_vectors_location, training_data_ids, words_to_ids):

    pretrained = dict()
    wordsList = []
    vocabulary = load_vocabulary(vocabulary_location)
    pretrained_embeddings = codecs.open(
        pretrained_vectors_location, 'rb', encoding)
    new_file = codecs.open("pretrained_file", "w", encoding)
    wordVectors = np.zeros([len(vocabulary), 300])

    print('Loading pretrained vectors....')
    count = 0
    for line in pretrained_embeddings:
        if line:
            # if count < 20:
            word = line.lstrip("„").lower().encode(encoding).rstrip().split()[
                0]
            new_file.write(word.decode(encoding))
            new_file.write("\n")
            if len(line.split()) > 1:
                if isinstance(line.split()[1], float):
                    vector = [float(i) for i in line.split()[1:]]
                    if word in words_to_ids: 
                        word_idx = words_to_ids[word]
                        wordVectors[word_idx] =  np.asarray(vector)
                        count += 1
                    vector = np.array(vector)

    return wordVectors

def pad_sequence(training_data):
    sequence_padded = []
    sequence_length = []

    max_length = max(map(lambda x : len(x), training_data))

    for data in training_data:
        data = list(data)
        data_ = data[:max_length] + [1]*max(max_length - len(data), 0)
        sequence_padded +=  [data_]
        sequence_length += [min(len(data), max_length)]
    # print(sequence_length)
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
    # print(words_to_ids)
    return words_to_ids, ids_to_words

def build_train_data(training_data, training_tags):

    training_data_ids = []
    training_tags_ids = []
    ids_data = []
    ids_tags = []
    words_to_ids, ids_to_words = convert_words_to_ids('./vocabulary/words')
    tags_to_ids, ids_to_tags = convert_words_to_ids('./vocabulary/tags')
    # print(tags_to_ids)
    # print(ids_to_words)

    for sentence_words, sentence_tags in zip(training_data, training_tags):
        # print('len sentence_words vs tags', len(sentence_words), len(sentence_tags))
        # print(sentence_tags)
        for word, tag in zip(sentence_words, sentence_tags):
            word = word.lower()
            
            if words_to_ids.get(word) != None:
                ids_data.append(words_to_ids[word])
            else:
                f = codecs.open("./results/test.txt", 'w', encoding)
                f.write(word)
                f.write('\n')
            if tags_to_ids.get(tag) != None:
                ids_tags.append(tags_to_ids.get(tag))
        training_data_ids.append(ids_data)
        ids_data = []
        training_tags_ids.append(ids_tags)
        ids_tags =[]

    # for sentence in training_data:
    #     print('sentence length', len(sentence))
    #     for word in sentence:
    #         if words_to_ids.get(word) != None:
    #             ids_data.append(words_to_ids[word])
    #     training_data_ids.append(ids_data)
    #     ids_data = []
    # for tags in training_tags:
    #     print('tags length', len(tags))
    #     for tag in tags:
    #         # print(tag)
    #         # print(tags_to_ids.get(tag))
    #         # print('===========')
    #         if tags_to_ids.get(tag) != None:
    #             ids_tags.append(tags_to_ids.get(tag))
    #     training_tags_ids.append(ids_tags)
    #     ids_tags =[]
    # print(training_tags_ids)
    return training_data_ids, training_tags_ids, words_to_ids, ids_to_words


def save_session(sess, saver, location):
    saver.save(sess, location)

def restore_session(location, saver=None, sess=None):
    print('Loading latest trained model')
    saver.restore(sess, location)

def predict_batch(sess, data, placeholders, parameters):
    # sess, saver = initialize_session()
    # restore_session('./results/model', saver, sess)

    fd, sequence_lengths = get_feed(data, placeholders,  lr=0.001, dropout=1.0)

    # get tag scores and transition params of CRF
    viterbi_sequences = []
    logits, trans_params = sess.run(
            [parameters['logits'], parameters['trans_params']], feed_dict=fd)

    # print(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print(logits[0].shape, sequence_lengths[0])
    # iterate over the sentences because no batching in vitervi_decode
    for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length] # keep only the valid steps
        # print('=========================================================')
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


def get_chunks(seq, tags_arr):
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
    tags = {}
    for i, tag in enumerate(tags_arr):
        # tag = tag.decode(encoding)
        tags[tag] = i
    # print(tags)
    default = tags['O']
    # print(tags[b'O'])
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
        # elif tok != default:
        #     tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
        #     if chunk_type is None:
        #         chunk_type, chunk_start = tok_chunk_type, i
        #     elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
        #         chunk = (chunk_type, chunk_start, i)
        #         chunks.append(chunk)
        #         chunk_type, chunk_start = tok_chunk_type, i
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
    
def evaluate(sess, test_data_ids, test_tags_id,test_id_to_words, placeholders, parameters):
    # batches_training_test, batches_tags_test = create_batches(test_data_ids, test_tags_id, n_test)
    print('I am in evaluate')

    accs = []
    tags = load_vocabulary_list('./vocabulary/tags')
    print(tags)

    correct_preds, total_correct, total_preds = 0., 0., 0.
    n = int(len(test_data_ids)/num_batches)
    correct = 0
    overall = 0
    O, b_loc, i_loc, b_org, i_org, b_pers, i_pers, b_oth, i_oth = 0, 0, 0, 0, 0, 0, 0, 0, 0 
    O_all, b_loc_all, i_loc_all, b_org_all, i_org_all, b_pers_all, i_pers_all, b_oth_all, i_oth_all = 0, 0, 0, 0, 0, 0, 0, 0, 0 
    batches_training, batches_tags = create_batches(test_data_ids, test_tags_id, 20)
    # print('OVerall number of batches', print(batches_tags))
    for i, (batch_words, batch_tags) in enumerate(zip(batches_training, batches_tags)):
        # print('Batch', i)

        labels_pred, sequence_lengths = predict_batch(sess, batch_words, placeholders, parameters)

        for lab, lab_pred, length, sentence in zip(batch_tags, labels_pred,
                                             sequence_lengths, batch_words):
    #         lab      = lab[:length]
    #         lab_pred = lab_pred[:length]
    #         accs    += [a==b for (a, b) in zip(lab, lab_pred)]

    #         lab_chunks      = set(get_chunks(lab, tags))
    #         lab_pred_chunks = set(get_chunks(lab_pred, tags))

    #         correct_preds += len(lab_chunks & lab_pred_chunks)
    #         total_preds   += len(lab_pred_chunks)
    #         total_correct += len(lab_chunks)
    # print("correct predictions", correct_preds)
    # print("total predictions", total_preds)
    # print("total correct", total_correct)
    # p   = correct_preds / total_preds if correct_preds > 0 else 0
    # r   = correct_preds / total_correct if correct_preds > 0 else 0
    # f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
    # acc = np.mean(accs)
    # print("acc", 100*acc, "f1", f1)
    # return f1
            # Go to build_train_data
            # =================================================
            lab      = lab[:length]
            lab_pred = lab_pred[:length]
            accs    += [a==b for (a, b) in zip(lab, lab_pred)]
            # print('Sentencec Ids')
            # print(sentence)
            # print('Real results')
            # print(lab)
            # print('Predicted results')
            # print(lab_pred)
            for a, b in zip(lab, lab_pred):
                # print(a, b)
                a_ = tags[a]
                overall += 1
                if a_ == 'O':
                    O_all += 1
                if a_ == 'B-Loc':
                    b_loc_all += 1
                if a_ =='I-Loc':
                    i_loc_all += 1
                if a_ =='B-Pers':
                    b_pers_all += 1
                if a_ =='I-Pers':
                    i_pers_all += 1
                if a_ =='B-Org':
                    b_org_all += 1
                if a_ =='I-Org':
                    i_org_all += 1
                if a_ =='B-Other':
                    b_oth_all += 1
                if a_ =='I-Other':
                    i_oth_all += 1
                if a == b:
                    if a_ != 'O':
                        correct += 1
                    if a_ =='O':
                        O += 1
                    if a_ =='B-Loc':
                        b_loc += 1
                    if a_ =='I-Loc':
                        i_loc += 1
                    if a_ =='B-Pers':
                        b_pers += 1
                    if a_ =='I-Pers':
                        i_pers += 1
                    if a_ =='B-Org':
                        b_org += 1
                    if a_ =='I-Org':
                        i_org += 1
                    if a_ =='B-Other':
                        b_oth += 1
                    if a_ =='I-Other':
                        i_oth += 1                 

    
    print(correct)
    print(overall)
    print('Overall result is:', correct * 100 / overall)
    print('Correct Other', O, O_all)
    print('Correct Loc', b_loc, b_loc_all,  i_loc,i_loc_all)
    print('Correct Org',  b_org,b_org_all,  i_org,i_org_all)
    print('Correct Pers',  b_pers,b_pers_all,   i_pers,i_pers_all)
    print('Correct B/I  Other',  b_oth,b_oth_all,  i_oth,i_oth_all)
    # print('This is result from labeeeeeeeeeeeeeeeeeeeeeeeeling')
    # print(batches_training[0][0])
    # print(labels_pred[0])
    # print('This is reeeeeeeeeeal label')
    # print(batches_tags[0][0])

def run_evaluate(location):
    training_data_ids, training_tags_ids, words_to_ids, ids_to_words = build_train_data(location)
    test_data_ids, test_tags_ids, _, _ = build_train_data(test_location)
    placeholders = define_placeholders()
    parameters = build_model(training_data_ids, words_to_ids, placeholders)
    sess, saver = initialize_session()
    n = int(len(training_data_ids)/num_batches)
    n_test = int(len(test_data_ids)/num_batches)

    batches_training, batches_tags = create_batches(training_data_ids, training_tags_ids, n)
    
    evaluate(test_data_ids, test_tags_id, placeholders, parameters)



def main():
    train_model()
    # run_evaluate('./dataset/bg/testb.txt')
    # get_fasttext_vocab(pretrained_vectors_location)
    # evaluate('./dataset/bg/testa.txt')

if __name__ == '__main__':
    main()





