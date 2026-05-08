import os
import sys

from src.rnn import train_and_test_model, MinCharLSTM


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AIRLINE_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'airline-passengers.csv')
SHAMPOO_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'shampoo.csv')
CNUS_CLEAN_PATH = os.path.join('data', 'sem2', 'lab3', 'cnus-clean.txt')

MIN_CHAR_LSTM_MODEL_PATH = os.path.join('models', 'min_char_lstm.npz')

OPTION = 'min-char-lstm'  # airline, shampoo, min-char-lstm
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

