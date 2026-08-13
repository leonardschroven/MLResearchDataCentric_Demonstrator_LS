"""Iterative data-selective training (follow-up study).

The companion paper ``Paper_DataSelectiveTraining`` analysed a **single** round of
the loop

    evaluate → identify weakspot → select data → retrain

and closed with three open threads that only a repeated loop can answer:

* whether the catastrophic-forgetting collapse it measured survives when the
  retraining set *accumulates* instead of being replaced each round;
* whether a **schedule** that relaxes the focus as the weak region shrinks
  ("expressible through either coverage control") beats a fixed setting;
* whether the operating point it read off a single round still holds once the
  weakspot is allowed to migrate.

This package implements that repeated loop. It is deliberately separate from
``scripts.dataselect`` (single round, frozen results) and ``scripts.weakspot``
(detection only) so those experiments and their swept CSVs keep working
unchanged; everything shared — the landscape, the selection kernel, the
detectors — is imported from them rather than copied.

Modules
-------
``models``  MLP architecture / optimiser variants and the per-iteration
            learning-rate schedules.
``loop``    the streamlit-free engine: one call runs a whole iterative
            experiment and returns every per-iteration metric.
``sweep``   resumable parameter sweep over the engine, one CSV row per
            (config × detector × iteration).
``plots``   plotly helpers shared by the experiment and visualisation pages.
"""
