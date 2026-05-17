import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rnn import train_and_test_model, MinCharLSTM, MinCharRNN

BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / 'artifacts' / 'sem2' / 'lab3'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data' / 'sem2' / 'lab3'

AIRLINE_DATA_PATH = str(DATA_DIR / 'airline-passengers.csv')
SHAMPOO_DATA_PATH = str(DATA_DIR / 'shampoo.csv')
INPUT_DATA_PATH = str(DATA_DIR / 'input.txt')
CNUS_CLEAN_PATH = str(DATA_DIR / 'cnus-clean.txt')


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_time_series() -> None:
    airline_params = {
        'num_epochs': 2000,
        'learning_rate': 0.01,
        'input_size': 1,
        'hidden_size': 2,
        'num_layers': 1,
        'num_classes': 1,
        'verbose': 100,
    }
    shampoo_params = {
        'num_epochs': 3500,
        'learning_rate': 0.0009,
        'input_size': 1,
        'hidden_size': 3,
        'num_layers': 1,
        'num_classes': 1,
        'verbose': 100,
    }

    print('\n=== AIRLINE ===')
    train_and_test_model(
        AIRLINE_DATA_PATH,
        'Airline Passengers Data',
        airline_params,
        plot_path=str(ASSETS_DIR / 'airline.png'),
    )

    print('\n=== SHAMPOO ===')
    train_and_test_model(
        SHAMPOO_DATA_PATH,
        'Shampoo Sales Data',
        shampoo_params,
        plot_path=str(ASSETS_DIR / 'shampoo.png'),
    )


def run_min_char_lstm(data_path: str, name: str, prompt: str, max_data: int = 30_000) -> None:
    data = Path(data_path).read_text()
    chars = list(set(data))
    model_path = MODELS_DIR / f'min_char_lstm_{name}.npz'
    plot_path = ASSETS_DIR / f'min_char_lstm_{name}.png'
    sample_path = ASSETS_DIR / f'min_char_lstm_{name}_sample.txt'

    params = {
        'hidden_size': 100,
        'seq_length': 16,
        'learning_rate': 0.1,
        'max_data': max_data,
    }

    print(f'\n=== MIN-CHAR-LSTM ({name}) ===')
    print('data has %d characters, %d unique.' % (len(data), len(chars)))
    model = MinCharLSTM(chars, params['hidden_size'], params['seq_length'], params['learning_rate'])
    model.train_and_test(data, params['max_data'], plot_path=str(plot_path))
    model.save_model(str(model_path))

    reloaded = MinCharLSTM(chars, params['hidden_size'], params['seq_length'], params['learning_rate'])
    reloaded.load_model(str(model_path))
    generated_text = reloaded.generate(prompt, n_chars=200)
    sample_path.write_text(f'Prompt: {prompt}\n\n{generated_text}\n')
    print('Sample saved to', sample_path)
    print(generated_text)


def run_min_char_rnn(data_path: str, name: str, prompt: str, max_data: int = 100_000) -> None:
    data = Path(data_path).read_text()
    model_path = MODELS_DIR / f'min_char_rnn_{name}.npz'
    plot_path = ASSETS_DIR / f'min_char_rnn_{name}.png'
    sample_path = ASSETS_DIR / f'min_char_rnn_{name}_sample.txt'

    print(f'\n=== MIN-CHAR-RNN ({name}) ===')
    print(f'data has {len(data)} characters, {len(set(data))} unique.')
    model = MinCharRNN(data, hidden_size=100, seq_length=16, learning_rate=0.1)
    model.train(max_data=max_data, verbose=200, plot_path=str(plot_path))
    model.save(str(model_path))

    reloaded = MinCharRNN(data, hidden_size=100, seq_length=16, learning_rate=0.1)
    reloaded.load(str(model_path))
    generated_text = reloaded.generate(prompt, n_chars=200)
    sample_path.write_text(f'Prompt: {prompt}\n\n{generated_text}\n')
    print('Sample saved to', sample_path)
    print(generated_text)


if __name__ == '__main__':
    ensure_dirs()
    run_time_series()
    run_min_char_rnn(INPUT_DATA_PATH, 'input', prompt='First Citizen:')
    run_min_char_rnn(CNUS_CLEAN_PATH, 'cnus_clean', prompt='the ')
    run_min_char_lstm(INPUT_DATA_PATH, 'input', prompt='First Citizen:')
    run_min_char_lstm(CNUS_CLEAN_PATH, 'cnus_clean', prompt='the ')
