from commons.optimizer import SGD
from commons.trainer import RnnlmTrainer
from commons.util import eval_perplexity
from datasets import ptb
from rnnlm import RNNLM

# set hyperparameters
batch_size = 20
wordvec_size = 100
hidden_size = 100  # number of elements in hideen layers of RNN
time_size = 35  # unfold size of RNN
lr = 20.0
max_epoch = 4
max_grad = 0.25

# read train dataset
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# generate model
model = RNNLM(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# train with gradient clipping
trainer.fit(xs, ts, max_epoch,
            batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# evaluate with test dataset
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('Test Perplexity: ', ppl_test)

# save parameters
model.save_params()
