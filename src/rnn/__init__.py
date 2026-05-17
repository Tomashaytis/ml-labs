try:
    from .lstm_torch import LSTM, load_data, sliding_windows, train_and_test_model
    _have_lstm = True
except Exception:
    LSTM = None
    load_data = None
    sliding_windows = None
    train_and_test_model = None
    _have_lstm = False

from .min_char_lstm import MinCharLSTM
from .min_char_rnn_class import MinCharRNN

__all__ = [
    'MinCharLSTM',
    'MinCharRNN',
]

if _have_lstm:
    __all__ = ['LSTM', 'load_data', 'sliding_windows', 'train_and_test_model'] + __all__