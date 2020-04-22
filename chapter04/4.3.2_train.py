from commons.util import create_contexts_target, to_cpu, to_gpu
from commons.optimizer import Adam
from commons.trainer import Trainer
from commons import config
from datasets import ptb
from skip_gram import SkipGram
from cbow import CBOW
import numpy as np
import pickle

# if train with GPU, unlock config.GPU
# config.GPU = True

# set hyperparameters
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# read dataset
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# init model
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# start train
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# save needed data for use after
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
