from __future__ import annotations

"""Evaluation: runs the vendored official/score.py, sacreBLEU (BLEU/chrF/chrF++),
optional COMET-22 (eval-only, never used for selection), bootstrap 95% CIs and
paired bootstrap A/B comparisons, reported by slice / length bucket / E-set. Spec §8.
"""
