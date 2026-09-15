"""Forecasting package — temporal state evolution S(t)->S(t+k)."""
from .state_builder import build_sequences  # noqa: F401
# CICIDS-2017 Dataset B exports
from .cicids_loader import load_all_cicids  # noqa: F401
from .cicids_state_builder import build_states, states_to_arrays  # noqa: F401
from .cicids_sequence import build_sequences as build_cicids_sequences, temporal_split  # noqa: F401
