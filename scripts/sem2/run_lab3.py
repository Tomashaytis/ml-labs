import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rnn import train_and_test_model, MinCharLSTM
from src.rnn import MinCharRNN

AIRLINE_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'airline-passengers.csv')
SHAMPOO_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'shampoo.csv')
CNUS_CLEAN_PATH = os.path.join('data', 'sem2', 'lab3', 'cnus-clean.txt')

MIN_CHAR_LSTM_MODEL_PATH = os.path.join('models', 'min_char_lstm.npz')
MIN_CHAR_RNN_MODEL_PATH = os.path.join('models', 'min_char_rnn_cnus_clean.npz')

OPTION = 'min-char-rnn'  # airline, shampoo, min-char-lstm, min-char-rnn
MODE = 'test'


if __name__ == "__main__":
    if OPTION == 'airline':
        params = {
            'num_epochs': 2000,
            'learning_rate': 0.01,
            'input_size': 1,
            'hidden_size': 2,
            'num_layers': 1,
            'num_classes': 1,
            'verbose': 100,
        }

        train_and_test_model(AIRLINE_DATA_PATH,'Airline Passengers Data', params)

    elif OPTION == 'shampoo':
        params = {
            'num_epochs': 3500,
            'learning_rate': 0.0009,
            'input_size': 1,
            'hidden_size': 3,
            'num_layers': 1,
            'num_classes': 1,
            'verbose': 100,
        }

        train_and_test_model(SHAMPOO_DATA_PATH,'Shampoo Sales Data', params)

    elif OPTION == 'min-char-lstm':
        with open(CNUS_CLEAN_PATH, 'r') as f:
            data = f.read()

        # All unique characters / entities in the data set.
        chars = list(set(data))

        # Hyperparameters.
        params = {
            'hidden_size': 100,        # Size of hidden state vectors; applies to h and c.
            'seq_length': 16,          # Number of steps to unroll the LSTM for
            'learning_rate': 0.1,      # Learning rate
            'max_data': 1000000,       # Stop when processed this much data
        }

        if MODE == 'train':
            print('data has %d characters, %d unique.' % (len(data), len(chars)))

            min_char_lstm = MinCharLSTM(
                chars,
                params['hidden_size'],
                params['seq_length'],
                params['learning_rate']
            )

            min_char_lstm.train_and_test(data, params['max_data'])

            min_char_lstm.save_model(MIN_CHAR_LSTM_MODEL_PATH)

        elif MODE == 'test':
            min_char_lstm = MinCharLSTM(
                chars,
                params['hidden_size'],
                params['seq_length'],
                params['learning_rate']
            )

            min_char_lstm.load_model(MIN_CHAR_LSTM_MODEL_PATH)

            prompt = input('Prompt: ')
            generated_text = min_char_lstm.generate(prompt, n_chars=100)

            print(f"LSTM:")
            print(generated_text)

    elif OPTION == 'min-char-rnn':
        with open(CNUS_CLEAN_PATH, 'r') as f:
            data = f.read()

        params = {
            'hidden_size': 100,
            'seq_length': 16,
            'learning_rate': 0.1,
        }

        rnn = MinCharRNN(data, hidden_size=params['hidden_size'], seq_length=params['seq_length'], learning_rate=params['learning_rate'])

        if MODE == 'train':
            print(f'data has {len(data)} characters, {len(rnn.chars)} unique.')
            rnn.train(max_data=100000, verbose=200)
            rnn.save(MIN_CHAR_RNN_MODEL_PATH)
        elif MODE == 'test':
            try:
                rnn.load(MIN_CHAR_RNN_MODEL_PATH)
            except Exception:
                print('No saved min-char-rnn model found; run with MODE=train first')
            else:
                prompt = input('Prompt: ')
                print('RNN:')
                print(rnn.generate(prompt, n_chars=200))

