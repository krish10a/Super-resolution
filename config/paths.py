"""Centralized project paths.

All paths resolve relative to this file's location (`config/`), so the project
can be moved or cloned anywhere without hardcoding `C:\Users\...` absolute paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_dir(*parts):
    return PROJECT_ROOT / 'data' / Path(*parts)


def raw_dir(*parts):
    return PROJECT_ROOT / 'data' / 'raw' / Path(*parts)


def pairs_dir(*parts):
    return PROJECT_ROOT / 'data' / 'pairs' / Path(*parts)


def pairs_real_dir(*parts):
    return PROJECT_ROOT / 'data' / 'pairs_real' / Path(*parts)


def checkpoints_dir(*parts):
    return PROJECT_ROOT / 'outputs' / 'checkpoints' / Path(*parts)


def outputs_dir(*parts):
    return PROJECT_ROOT / 'outputs' / Path(*parts)


__all__ = [
    'PROJECT_ROOT', 'data_dir', 'raw_dir', 'pairs_dir', 'pairs_real_dir',
    'checkpoints_dir', 'outputs_dir',
]
