from __future__ import annotations

"""Data pipeline: download opus-100 (en-fr), normalize, filter, dedupe, run the
leakage guard against dev/test/E-sets, and build the held-out eval proxies E1/E2/E3.
Logs filter counts to W&B and to data_manifest.json. Spec §3.
"""
