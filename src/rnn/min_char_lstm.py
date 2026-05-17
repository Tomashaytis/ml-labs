"""
Minimal character-based language model learning with an LSTM architecture.

Overall code structure based on Andrej Karpathy's min-char-rnn model:
    https://gist.github.com/karpathy/d4dee566867f8291f086

But the architecture is modified to be LSTM rather than vanilla RNN.
he companion blog post is:
    https://eli.thegreenplace.net/2018/minimal-character-based-lstm-implementation/

Tested with Python 3.6

Eli Bendersky [https://eli.thegreenplace.net]
BSD License per original (@karpathy)
"""

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np


class MinCharLSTM:
    def __init__(self, chars: list, hidden_size: int = 100, seq_length: int = 16, learning_rate: float = 0.1):
        # Each character in the vocabulary gets a unique integer index assigned, in the
        # half-open interval [0:N). These indices are useful to create one-hot encoded
        # vectors that represent characters in numerical computations.
        self._chars = chars
        self._char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self._ix_to_char = {i: ch for i, ch in enumerate(chars)}

        self._V = len(chars)
        self._H = hidden_size  # Size of hidden state vectors; applies to h and c.
        self._seq_length = seq_length  # number of steps to unroll the LSTM for
        self._learning_rate = learning_rate

        # The input x is concatenated with state h, and the joined vector is used to
        # feed into most blocks within the LSTM cell. The combined height of the column
        # vector is HV.
        self._HV = self._H + self._V

        # Model parameters/weights -- these are shared among all steps. Weights
        # initialized randomly; biases initialized to 0.
        # Inputs are characters one-hot encoded in a vocab-sized vector.
        # Dimensions: H = hidden_size, V = vocab_size, HV = hidden_size + vocab_size
        self._Wf = np.random.randn(self._H, self._HV) * 0.01
        self._bf = np.zeros((self._H, 1))
        self._Wi = np.random.randn(self._H, self._HV) * 0.01
        self._bi = np.zeros((self._H, 1))
        self._Wcc = np.random.randn(self._H, self._HV) * 0.01
        self._bcc = np.zeros((self._H, 1))
        self._Wo = np.random.randn(self._H, self._HV) * 0.01
        self._bo = np.zeros((self._H, 1))
        self._Wy = np.random.randn(self._V, self._H) * 0.01
        self._by = np.zeros((self._V, 1))

        # Memory variables for Adagrad.
        self._mWf = np.zeros_like(self._Wf)
        self._mbf = np.zeros_like(self._bf)
        self._mWi = np.zeros_like(self._Wi)
        self._mbi = np.zeros_like(self._bi)
        self._mWcc = np.zeros_like(self._Wcc)
        self._mbcc = np.zeros_like(self._bcc)
        self._mWo = np.zeros_like(self._Wo)
        self._mbo = np.zeros_like(self._bo)
        self._mWy = np.zeros_like(self._Wy)
        self._mby = np.zeros_like(self._by)

    @staticmethod
    def sigmoid(z):
        """
        Computes sigmoid function.

        z: array of input values.

        Returns array of outputs, sigmoid(z).
        """
        # Note: this version of sigmoid tries to avoid overflows in the computation
        # of e^(-z), by using an alternative formulation when z is negative, to get
        # 0. e^z / (1+e^z) is equivalent to the definition of sigmoid, but we won't
        # get e^(-z) to overflow when z is very negative.
        # Since both the x and y arguments to np.where are evaluated by Python, we
        # may still get overflow warnings for large z elements; therefore we ignore
        # warnings during this computation.
        with np.errstate(over='ignore', invalid='ignore'):
            return np.where(
                z >= 0,
                1 / (1 + np.exp(-z)),
                np.exp(z) / (1 + np.exp(z))
            )

    def loss_function(self, inputs, targets, hprev, cprev):
        """
        Runs forward and backward passes through the RNN.

        inputs, targets: Lists of integers. For some i, inputs[i] is the input
                           character (encoded as an index into the ix_to_char map)
                           and targets[i] is the corresponding next character in the
                           training data (similarly encoded).
        hprev: Hx1 array of initial hidden state
        cprev: Hx1 array of initial hidden state

        returns: loss, gradients on model parameters, and last hidden states
        """
        # Caches that keep values computed in the forward pass at each time step, to
        # be reused in the backward pass.
        xs, xhs, ys, ps, hs, cs, fgs, igs, ccs, ogs = (
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {})

        # Initial incoming states.
        hs[-1] = np.copy(hprev)
        cs[-1] = np.copy(cprev)

        loss = 0
        # Forward pass
        for t in range(len(inputs)):
            # Input at time step t is xs[t]. Prepare a one-hot encoded vector of
            # shape (V, 1). inputs[t] is the index where the 1 goes.
            xs[t] = np.zeros((self._V, 1))
            xs[t][inputs[t]] = 1

            # hprev and xs[t] are column vector; stack them together into a "taller"
            # column vector - first the elements of x, then h.
            xhs[t] = np.vstack((xs[t], hs[t - 1]))

            # Gates f, i and o.
            fgs[t] = self.sigmoid(np.dot(self._Wf, xhs[t]) + self._bf)
            igs[t] = self.sigmoid(np.dot(self._Wi, xhs[t]) + self._bi)
            ogs[t] = self.sigmoid(np.dot(self._Wo, xhs[t]) + self._bo)

            # Candidate cc.
            ccs[t] = np.tanh(np.dot(self._Wcc, xhs[t]) + self._bcc)

            # This step's h and c.
            cs[t] = fgs[t] * cs[t - 1] + igs[t] * ccs[t]
            hs[t] = np.tanh(cs[t]) * ogs[t]

            # Softmax for output.
            ys[t] = np.dot(self._Wy, hs[t]) + self._by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

            # Cross-entropy loss.
            loss += -np.log(ps[t][targets[t], 0])

        # Initialize gradients of all weights/biases to 0.
        dWf = np.zeros_like(self._Wf)
        dbf = np.zeros_like(self._bf)
        dWi = np.zeros_like(self._Wi)
        dbi = np.zeros_like(self._bi)
        dWcc = np.zeros_like(self._Wcc)
        dbcc = np.zeros_like(self._bcc)
        dWo = np.zeros_like(self._Wo)
        dbo = np.zeros_like(self._bo)
        dWy = np.zeros_like(self._Wy)
        dby = np.zeros_like(self._by)

        # Incoming gradients for h and c; for backwards loop step these represent
        # dh[t] and dc[t]; we do truncated BPTT, so assume they are 0 initially.
        dhnext = np.zeros_like(hs[0])
        dcnext = np.zeros_like(cs[0])

        # The backwards pass iterates over the input sequence backwards.
        for t in reversed(range(len(inputs))):
            # Backprop through the gradients of loss and softmax.
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            # Compute gradients for the Wy and by parameters.
            dWy += np.dot(dy, hs[t].T)
            dby += dy

            # Backprop through the fully-connected layer (Wy, by) to h. Also add up
            # the incoming gradient for h from the next cell.
            dh = np.dot(self._Wy.T, dy) + dhnext

            # Backprop through multiplication with output gate; here "dtanh" means
            # the gradient at the output of tanh.
            dctanh = ogs[t] * dh
            # Backprop through the tanh function; since cs[t] branches in two
            # directions we add dcnext too.
            dc = dctanh * (1 - np.tanh(cs[t]) ** 2) + dcnext

            # Backprop through multiplication with the tanh; here "dhogs" means
            # the gradient at the output of the sigmoid of the output gate. Then
            # backprop through the sigmoid itself (ogs[t] is the sigmoid output).
            dhogs = dh * np.tanh(cs[t])
            dho = dhogs * ogs[t] * (1 - ogs[t])

            # Compute gradients for the output gate parameters.
            dWo += np.dot(dho, xhs[t].T)
            dbo += dho

            # Backprop dho to the xh input.
            dxh_from_o = np.dot(self._Wo.T, dho)

            # Backprop through the forget gate: sigmoid and elementwise mul.
            dhf = cs[t - 1] * dc * fgs[t] * (1 - fgs[t])
            dWf += np.dot(dhf, xhs[t].T)
            dbf += dhf
            dxh_from_f = np.dot(self._Wf.T, dhf)

            # Backprop through the input gate: sigmoid and elementwise mul.
            dhi = ccs[t] * dc * igs[t] * (1 - igs[t])
            dWi += np.dot(dhi, xhs[t].T)
            dbi += dhi
            dxh_from_i = np.dot(self._Wi.T, dhi)

            dhcc = igs[t] * dc * (1 - ccs[t] ** 2)
            dWcc += np.dot(dhcc, xhs[t].T)
            dbcc += dhcc
            dxh_from_cc = np.dot(self._Wcc.T, dhcc)

            # Combine all contributions to dxh, and extract the gradient for the
            # h part to propagate backwards as dhnext.
            dxh = dxh_from_o + dxh_from_f + dxh_from_i + dxh_from_cc
            dhnext = dxh[self._V:, :]

            # dcnext from dc and the forget gate.
            dcnext = fgs[t] * dc

        # Gradient clipping to the range [-5, 5].
        for dparam in [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return (loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
                hs[len(inputs) - 1], cs[len(inputs) - 1])

    def sample(self, h, c, seed_ix, n):
        """
        Sample a sequence of integers from the model.

        Runs the LSTM in forward mode for n steps; seed_ix is the seed letter for
        the first time step, h and c are the memory state. Returns a sequence of
        letters produced by the model (indices).
        """
        x = np.zeros((self._V, 1))
        x[seed_ix] = 1
        ixes = []

        for t in range(n):
            # Run the forward pass only.
            xh = np.vstack((x, h))
            fg = self.sigmoid(np.dot(self._Wf, xh) + self._bf)
            ig = self.sigmoid(np.dot(self._Wi, xh) + self._bi)
            og = self.sigmoid(np.dot(self._Wo, xh) + self._bo)
            cc = np.tanh(np.dot(self._Wcc, xh) + self._bcc)
            c = fg * c + ig * cc
            h = np.tanh(c) * og
            y = np.dot(self._Wy, h) + self._by
            p = np.exp(y) / np.sum(np.exp(y))

            # Sample from the distribution produced by softmax.
            ix = np.random.choice(range(self._V), p=p.ravel())
            x = np.zeros((self._V, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def grad_check(self, inputs, targets, hprev, cprev):
        """Gradient check for LSTM model."""
        num_checks, delta = 10, 1e-5
        (_, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
         _, _) = self.loss_function(inputs, targets, hprev, cprev)
        for param, dparam, name in zip(
                [self._Wf, self._bf, self._Wi, self._bi, self._Wcc, self._bcc, self._Wo, self._bo, self._Wy, self._by],
                [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby],
                ['Wf', 'bf', 'Wi', 'bi', 'Wcc', 'bcc', 'Wo', 'bo', 'Wy', 'by']):
            assert dparam.shape == param.shape
            print(name)
            for i in range(num_checks):
                ri = np.random.randint(0, param.size)
                old_val = param.flat[ri]
                param.flat[ri] = old_val + delta
                numloss0 = self.loss_function(inputs, targets, hprev, cprev)[0]
                param.flat[ri] = old_val - delta
                numloss1 = self.loss_function(inputs, targets, hprev, cprev)[0]
                param.flat[ri] = old_val  # reset
                grad_analytic = dparam.flat[ri]
                grad_numerical = (numloss0 - numloss1) / (2 * delta)
                if grad_numerical + grad_analytic == 0:
                    rel_error = 0
                else:
                    rel_error = (abs(grad_analytic - grad_numerical) /
                                 abs(grad_numerical + grad_analytic))
                print('%s, %s => %e' % (grad_numerical, grad_analytic, rel_error))

    def basic_grad_check(self, data):
        """Basic gradient checking."""
        inputs = [self._char_to_ix[ch] for ch in data[:self._seq_length]]
        targets = [self._char_to_ix[ch] for ch in data[1:self._seq_length + 1]]
        hprev = np.random.randn(self._H, 1)
        cprev = np.random.randn(self._H, 1)
        self.grad_check(inputs, targets, hprev, cprev)

    def train_and_test(self, data, max_data, plot_path=None):
        """Train model and test for symbols periodically."""
        start = time.time()

        iterations = []
        losses = []

        smooth_loss = -np.log(1.0 / self._V) * self._seq_length

        n, p = 0, 0

        while p < max_data:
            # Prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self._seq_length + 1 >= len(data) or n == 0:
                # Reset RNN memory
                hprev = np.zeros((self._H, 1))
                cprev = np.zeros((self._H, 1))
                p = 0  # go from start of data

            # In each step we unroll the RNN for seq_length cells, and present it with
            # seq_length inputs and seq_length target outputs to learn.
            inputs = [self._char_to_ix[ch] for ch in data[p:p + self._seq_length]]
            targets = [self._char_to_ix[ch] for ch in data[p + 1:p + self._seq_length + 1]]

            # Sample from the model now and then.
            if n % 1000 == 0:
                sample_ix = self.sample(hprev, cprev, inputs[0], 200)
                print('---- \n Starting symbol >>> ', self._ix_to_char[inputs[0]])
                txt = ''.join(self._ix_to_char[ix] for ix in sample_ix)
                print('Generated text after symbol \n %s \n ----' % (txt,))

            # Forward seq_length characters through the RNN and fetch gradient.
            (loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
             hprev, cprev) = self.loss_function(inputs, targets, hprev, cprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            iterations.append(n)
            losses.append(smooth_loss)

            if n % 200 == 0:
                print('iter %d (p=%d), loss %f' % (n, p, smooth_loss))

            # Perform parameter update with Adagrad.
            for param, dparam, mem in zip(
                    [self._Wf, self._bf, self._Wi, self._bi, self._Wcc, self._bcc, self._Wo, self._bo, self._Wy, self._by],
                    [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby],
                    [self._mWf, self._mbf, self._mWi, self._mbi, self._mWcc, self._mbcc, self._mWo, self._mbo, self._mWy, self._mby]):
                mem += dparam * dparam
                param += -self._learning_rate * dparam / np.sqrt(mem + 1e-8)

            p += self._seq_length
            n += 1

        end = time.time()
        time_delta = end - start
        print('Time taken %.2f seconds.' %time_delta)

        plt.plot(iterations, losses)
        plt.title('Loss per iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        if plot_path:
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate(self, prompt: str, n_chars=200, temperature=1.0):
        """Generate text for prompt."""
        if not prompt:
            return ""

        for ch in prompt:
            if ch not in self._char_to_ix:
                raise ValueError(f"Character '{ch}' not in vocabulary")

        # Start state
        h = np.zeros((self._H, 1))
        c = np.zeros((self._H, 1))

        # Handle prompt symbols without generation
        for ch in prompt:
            ix = self._char_to_ix[ch]
            x = np.zeros((self._V, 1))
            x[ix] = 1

            xh = np.vstack((x, h))
            fg = self.sigmoid(np.dot(self._Wf, xh) + self._bf)
            ig = self.sigmoid(np.dot(self._Wi, xh) + self._bi)
            og = self.sigmoid(np.dot(self._Wo, xh) + self._bo)
            cc = np.tanh(np.dot(self._Wcc, xh) + self._bcc)
            c = fg * c + ig * cc
            h = np.tanh(c) * og

        x = np.zeros((self._V, 1))
        x[self._char_to_ix[prompt[-1]]] = 1

        # Generate text
        generated_ixes = []
        for _ in range(n_chars):
            xh = np.vstack((x, h))
            fg = self.sigmoid(np.dot(self._Wf, xh) + self._bf)
            ig = self.sigmoid(np.dot(self._Wi, xh) + self._bi)
            og = self.sigmoid(np.dot(self._Wo, xh) + self._bo)
            cc = np.tanh(np.dot(self._Wcc, xh) + self._bcc)
            c = fg * c + ig * cc
            h = np.tanh(c) * og
            y = np.dot(self._Wy, h) + self._by

            p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
            ix = np.random.choice(range(self._V), p=p.ravel())

            x = np.zeros((self._V, 1))
            x[ix] = 1
            generated_ixes.append(ix)

        generated_text = ''.join(self._ix_to_char[ix] for ix in generated_ixes)

        return generated_text

    def save_model(self, filepath):
        """Save model parameters to file."""
        np.savez(
            filepath,
            # Основные параметры
            Wf=self._Wf, bf=self._bf,
            Wi=self._Wi, bi=self._bi,
            Wcc=self._Wcc, bcc=self._bcc,
            Wo=self._Wo, bo=self._bo,
            Wy=self._Wy, by=self._by,
            # Adagrad memory variables
            mWf=self._mWf, mbf=self._mbf,
            mWi=self._mWi, mbi=self._mbi,
            mWcc=self._mWcc, mbcc=self._mbcc,
            mWo=self._mWo, mbo=self._mbo,
            mWy=self._mWy, mby=self._mby,
            # Параметры архитектуры
            vocab_size=self._V,
            hidden_size=self._H,
            seq_length=self._seq_length,
            learning_rate=self._learning_rate,
            chars = np.array(self._chars)
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters from file."""
        data = np.load(filepath, allow_pickle=True)

        # Загрузка основных параметров
        self._Wf = data['Wf']
        self._bf = data['bf']
        self._Wi = data['Wi']
        self._bi = data['bi']
        self._Wcc = data['Wcc']
        self._bcc = data['bcc']
        self._Wo = data['Wo']
        self._bo = data['bo']
        self._Wy = data['Wy']
        self._by = data['by']

        # Загрузка Adagrad memory variables
        self._mWf = data['mWf']
        self._mbf = data['mbf']
        self._mWi = data['mWi']
        self._mbi = data['mbi']
        self._mWcc = data['mWcc']
        self._mbcc = data['mbcc']
        self._mWo = data['mWo']
        self._mbo = data['mbo']
        self._mWy = data['mWy']
        self._mby = data['mby']

        self._chars = data['chars']
        self._char_to_ix = {ch: i for i, ch in enumerate(self._chars)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._chars)}

        # Обновление параметров архитектуры (если нужно)
        if 'vocab_size' in data:
            self._V = int(data['vocab_size'])
        if 'hidden_size' in data:
            self._H = int(data['hidden_size'])
            self._HV = self._H + self._V  # Пересчитываем HV
        if 'seq_length' in data:
            self._seq_length = int(data['seq_length'])
        if 'learning_rate' in data:
            self._learning_rate = float(data['learning_rate'])

        print(f"Model loaded from {filepath}")
        print(f"  Vocab size: {self._V}, Hidden size: {self._H}")
        print(f"  Seq length: {self._seq_length}, Learning rate: {self._learning_rate}")
