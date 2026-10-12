"""SearchJev: training of a calibrated System-1 decision model on Qwen3.5 backbones (JEV row format).

The package is split into light-weight, torch-free modules (``labels``,
``schema``, ``prompting``, ``config``, ``metrics``) and heavier modules that
import torch/transformers/peft (``data``, ``collator``, ``modeling``, ``loss``,
``trainer``, ``calibration``, ``inference``).  Importing the package itself
never imports torch, so the light modules can be used in data-preparation
processes without CUDA initialisation.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
