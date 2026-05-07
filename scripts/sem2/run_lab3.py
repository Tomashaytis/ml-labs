import os
import sys
from src.rnn import train_and_test_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AIRLINE_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'airline-passengers.csv')
SHAMPOO_DATA_PATH = os.path.join('data', 'sem2', 'lab3', 'shampoo.csv')

OPTION = 'shampoo'  # airline, shampoo


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
