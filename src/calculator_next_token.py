#!/usr/bin/env python3
"""Compatibility entry point for the calculator next-token experiment.

The implementation lives in:
- src.data for deterministic integer equation generation and tokenization
- src.model for the NumPy transformer
- src.train for training and artifact saving
- src.cli_calculator for inference
"""

from src.train import main


if __name__ == "__main__":
    main()
