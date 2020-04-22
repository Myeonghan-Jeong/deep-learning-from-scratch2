from negative_sampling_layer import NegativeSamplingLoss
from commons.layers import Embedding
from commons.np import *


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # init weights
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # init layers
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)  # use Embedding layer
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(
            W_out, corpus, power=0.75, sample_size=5)

        # collect all params and grads in list
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # save dispersion representation of word in instance variable
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
