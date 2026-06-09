"""Shared setup for the experiment scripts: matplotlib backend and repo paths.

Run experiments via ``python run.py <name>`` from the repo root.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow use("Agg"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/_common.py -> repo root
MODELS = os.path.join(ROOT, "models")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
WEIGHTS = os.path.join(MODELS, "gpt2_pytorch_model.bin")
