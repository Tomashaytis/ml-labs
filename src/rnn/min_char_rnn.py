"""
Minimal character-level RNN wrapped in a class for reuse.
Ported from min_char_rnn.py (Karpathy) and adapted to class API.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


class MinCharRNN:
    def __init__(self, data: str, hidden_size: int = 100, seq_length: int = 16, learning_rate: float = 0.1):
        self.data = data
        self.chars = list(set(data))
        self.data_size = len(data)
        self.vocab_size = len(self.chars)

        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # parameters
        V = self.vocab_size
        H = self.hidden_size
        self.Wxh = np.random.randn(H, V) * 0.01
        self.Whh = np.random.randn(H, H) * 0.01
        self.Why = np.random.randn(V, H) * 0.01
        self.bh = np.zeros((H, 1))
        self.by = np.zeros((V, 1))

        # adagrad memory
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def loss_function(self, inputs: List[int], targets: List[int], hprev: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0.0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])

        # gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(self, h: np.ndarray, seed_ix: int, n: int) -> List[int]:
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def train(self, max_data: int = 1000000, verbose: int = 200, plot_path: str | None = None):
        n, p = 0, 0
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length
        iterations = []
        losses = []

        while p < max_data:
            if p + self.seq_length + 1 >= len(self.data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))
                p = 0

            inputs = [self.char_to_ix[ch] for ch in self.data[p:p + self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1:p + self.seq_length + 1]]

            if n % 1000 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print('---- \n Starting symbol >>> ', self.ix_to_char[inputs[0]])
                print('Generated text after symbol \n %s \n ----' % (txt,))

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.loss_function(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            iterations.append(n)
            losses.append(smooth_loss)

            if n % verbose == 0:
                print('iter %d (p=%d), loss: %f' % (n, p, smooth_loss))

            for param, dparam, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [dWxh, dWhh, dWhy, dbh, dby],
                [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
            ):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            p += self.seq_length
            n += 1

        if plot_path:
            import matplotlib.pyplot as plt

            plt.plot(iterations, losses)
            plt.title('Loss per iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()

    def generate(self, seed: str, n_chars: int = 200) -> str:
        if not seed:
            return ''
        for ch in seed:
            if ch not in self.char_to_ix:
                raise ValueError(f"Character '{ch}' not in vocabulary")
        h = np.zeros((self.hidden_size, 1))
        for ch in seed[:-1]:
            ix = self.char_to_ix[ch]
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        seed_ix = self.char_to_ix[seed[-1]]
        gen_ix = self.sample(h, seed_ix, n_chars)
        return ''.join(self.ix_to_char[ix] for ix in gen_ix)

    def save(self, filepath: str):
        np.savez(filepath, Wxh=self.Wxh, Whh=self.Whh, Why=self.Why, bh=self.bh, by=self.by,
                 mWxh=self.mWxh, mWhh=self.mWhh, mWhy=self.mWhy, mbh=self.mbh, mby=self.mby,
                 vocab=np.array(self.chars), hidden_size=self.hidden_size, seq_length=self.seq_length,
                 learning_rate=self.learning_rate)

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.Wxh = data['Wxh']
        self.Whh = data['Whh']
        self.Why = data['Why']
        self.bh = data['bh']
        self.by = data['by']
        self.mWxh = data['mWxh']
        self.mWhh = data['mWhh']
        self.mWhy = data['mWhy']
        self.mbh = data['mbh']
        self.mby = data['mby']
        self.chars = list(data['vocab'])
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
