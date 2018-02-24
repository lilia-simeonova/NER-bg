encoding = 'utf-8'

vocabulary_location = './vocabulary/words-bg'
vocabulary_tags_location = './vocabulary/tags-bg'
vocabulary_chars_location = './vocabulary/chars-bg'
model = './results/model-bg-4'

# vocabulary_location = './vocabulary/words'
# vocabulary_tags_location = './vocabulary/tags'
# vocabulary_chars_location = './vocabulary/chars'
# model = './results/model-en'

# train_location = './dataset/bg/random1.txt'
# dev_location = './dataset/bg/random2.txt'
# test_location = './dataset/bg/random2.txt'

all_data = './dataset/bg/all_data'

train_location = './dataset/bg/all_data/cross/train_2.txt'
dev_location = './dataset/bg/all_data/cross/test_2.txt'
test_location = './dataset/bg/all_data/cross/test_2.txt'

result_location = "./results/tagged/res_2_1"

pretrained_vectors_location = './embeddings/wiki.bg.vec'
trimmed_embeddings_file = 'embeddings/trimmed.npz'

# pretrained_vectors_location = './embeddings/glove.6B.300d.txt'
# trimmed_embeddings_file = 'embeddings/en-trimmed.npz'

_lr_m = 'adam'
dropout = 0.5
lr = 0.001
lr_decay = 0.9
hidden_size_lstm = 300
hidden_size_char = 100
dim = 300
dim_char = 100
ntags = 9


batches_size = 20
nepoch_no_imprv = 5

epoch_range = 15
