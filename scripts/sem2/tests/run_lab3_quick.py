import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rnn import train_and_test_model

AIRLINE_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'airline-passengers.csv')
SHAMPOO_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'shampoo.csv')

def quick_run():
    airline_params = {
        'num_epochs': 10,
        'learning_rate': 0.01,
        'input_size': 1,
        'hidden_size': 2,
        'num_layers': 1,
        'num_classes': 1,
        'verbose': 5,
    }

    shampoo_params = {
        'num_epochs': 10,
        'learning_rate': 0.0009,
        'input_size': 1,
        'hidden_size': 3,
        'num_layers': 1,
        'num_classes': 1,
        'verbose': 5,
    }

    print('Running airline quick test')
    train_and_test_model(AIRLINE_DATA_PATH, 'Airline Quick', airline_params)

    print('Running shampoo quick test')
    train_and_test_model(SHAMPOO_DATA_PATH, 'Shampoo Quick', shampoo_params)


if __name__ == '__main__':
    quick_run()
