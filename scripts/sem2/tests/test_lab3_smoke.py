import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rnn import MinCharLSTM, MinCharRNN

CNUS_CLEAN_PATH = os.path.join('data', 'sem2', 'lab3', 'cnus-clean.txt')

def read_sample(path, max_chars=2000):
    with open(path, 'r') as f:
        data = f.read()
    return data[:max_chars]


def smoke():
    data = read_sample(CNUS_CLEAN_PATH)
    chars = list(set(data))

    print('Data length:', len(data), 'Unique chars:', len(chars))

    lstm = MinCharLSTM(chars, hidden_size=20, seq_length=8, learning_rate=0.1)
    print('LSTM generate:', lstm.generate('a', n_chars=50))

    rnn = MinCharRNN(data, hidden_size=20, seq_length=8, learning_rate=0.1)
    print('RNN generate:', rnn.generate('a', n_chars=50))


if __name__ == '__main__':
    smoke()
