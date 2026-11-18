"""Automated Study 1 metrics for 2x2 four-view grids.

The pure helpers in :mod:`evaluation.metrics.views` and :mod:`evaluation.metrics.stats`
are CPU-only and imported by the tests. Model-backed scorers (masks, DINOv2,
DreamSim, Orient Anything V2) import torch lazily so the package stays importable
in the lightweight CI environment.
"""
