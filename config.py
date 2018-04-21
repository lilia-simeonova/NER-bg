encoding = 'utf-8'

vocabulary_location = './vocabulary/words-bg'
vocabulary_tags_location = './vocabulary/tags-bg'
vocabulary_chars_location = './vocabulary/chars-bg'
model = './results/models/new_res'

# vocabulary_location = './vocabulary/words'
# vocabulary_tags_location = './vocabulary/tags'
# vocabulary_chars_location = './vocabulary/chars'
# model = './results/model-en'

# train_location = './dataset/en/train.txt'
# dev_location = './dataset/en/testb.txt'
# test_location = './dataset/en/testa.txt'

all_data = './dataset/bg/all_data'

train_location = './dataset/train_0.txt'
dev_location = './dataset/test_0.txt'
test_location = './dataset/test_0.txt'

# train_location = './dataset-en/train.txt'
# dev_location = './dataset-en/testb.txt'
# test_location = './dataset-en/testa.txt'

result_location = "./results/tagged/new_res"

pretrained_vectors_location = './embeddings/wiki.bg.vec'
trimmed_embeddings_file = 'embeddings/trimmed.npz'

# pretrained_vectors_location = './embeddings/glove.6B.300d.txt'
# trimmed_embeddings_file = 'embeddings/trimmed-en.npz'

# pretrained_vectors_location = './embeddings/glove.6B.300d.txt'
# trimmed_embeddings_file = 'embeddings/en-trimmed.npz'

_lr_m = 'adam'
dropout = 2
lr = 0.01
lr_decay = 0.5
hidden_size_lstm = 300
hidden_size_char = 150
dim = 300
dim_char = 150
ntags = 9
clip = 1

# may want to change the emb size of chars to 100 in both places

batches_size = 15
nepoch_no_imprv = 50

epoch_range = 100