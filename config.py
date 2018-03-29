encoding = 'utf-8'

vocabulary_location = './vocabulary/words-bg'
vocabulary_tags_location = './vocabulary/tags-bg'
vocabulary_chars_location = './vocabulary/chars-bg'
model = './results/models/res_bg_0_30epoch'

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

result_location = "./results/tagged/res_bg_0_30epoch"

pretrained_vectors_location = './embeddings/wiki.bg.vec'
trimmed_embeddings_file = 'embeddings/trimmed.npz'

# pretrained_vectors_location = './embeddings/glove.6B.300d.txt'
# trimmed_embeddings_file = 'embeddings/en-trimmed.npz'

_lr_m = 'adam'
dropout = 0.5
lr = 0.001
lr_decay = 1
hidden_size_lstm = 300
hidden_size_char = 100
dim = 300
dim_char = 100
ntags = 9
clip = -1

batches_size = 5
nepoch_no_imprv = 50

epoch_range = 100