"""Claim-level faithfulness auditing (O4 / RQ2).

Separate from ``nla.reconstructor`` on purpose. Reconstruction fidelity and
claim-level faithfulness are different properties, and keeping them in different
modules that report into different tables is how this codebase keeps from
quietly conflating them.
"""
