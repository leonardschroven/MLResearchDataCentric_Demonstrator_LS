"""Data-selective training experiment.

Builds on the weakspot-identification methods in ``scripts.weakspot`` and adds a
training loop that (1) trains a model on a dataset with an induced weakspot,
(2) identifies where the model is weak, (3) selects new labelled data around that
weakspot via a Gaussian selection kernel, and (4) retrains and re-evaluates.

Kept deliberately separate from ``scripts.weakspot`` so the existing Weakspot
Experiment / Visualise-Results pages keep working unchanged.
"""
