"""Comparison signals for the primary research question (O3).

Three signals, all cheaper than running an autoencoder:

* ``convergence`` - agreement between parsed intermediate answers (Liu & Wang)
* ``probe`` - a linear correctness probe on hidden states (Zhang et al.)
* ``semantic_entropy`` - meaning-level uncertainty (Farquhar et al.)

The study is set up so that any of these beating the verbalised readout is a
reportable result, not a failure. Project_Review_II.md section 1.3 says so
explicitly.
"""
