from __future__ import annotations

# Offline unit tests for scripts/gap_v2.py and scripts/gap_v2_stats.py (EXPLORATORY / POST-HOC
# analysis). Tiny fixtures only: no HF access, no pulled model (the NLL test uses a tiny
# random-init model), the committed tokenizer/spm.model for tokenisation.
import itertools
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm

from scripts import gap_v2 as g
from scripts import gap_v2_stats as st

REPO_ROOT = Path(__file__).resolve().parents[1]


def _walsh(k: int, n: int = 16) -> np.ndarray:
    return np.array([(-1.0) ** bin(i & k).count("1") for i in range(n)])


def _toy() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """16 rows, E1 = rows 0-7, E3 = rows 8-15. x1 and x2 are Walsh columns (mutually orthogonal and
    orthogonal to D) plus a domain shift, so the Shapley answer is known in closed form:
    y = 3*x1 + 5*x2 + 4*D, delta1 = xbar_E1 - xbar_E3 = -2, delta2 = +1  ->
    phi1 = 3 * -2 = -6, phi2 = 5 * 1 = 5, gap = -5, residual = -gamma = -4."""
    domain = (1.0 - _walsh(8)) / 2.0
    x1 = _walsh(1) + 2.0 * domain
    x2 = _walsh(2) - 1.0 * domain
    y = 3.0 * x1 + 5.0 * x2 + 4.0 * domain
    return y, {"g1": x1.reshape(-1, 1), "g2": x2.reshape(-1, 1)}, domain


def test_shapley_known_answer_and_sum_identity() -> None:
    y, groups, domain = _toy()
    res = st.decompose(y, groups, domain)
    assert res["gap"] == pytest.approx(-5.0)
    assert res["contribution"]["g1"] == pytest.approx(-6.0)
    assert res["contribution"]["g2"] == pytest.approx(5.0)
    assert res["residual"] == pytest.approx(-4.0)
    # shares sum to the total explained part EXACTLY, and with the residual to the gap
    assert sum(res["contribution"].values()) == pytest.approx(res["explained_total"], abs=1e-12)
    assert sum(res["contribution"].values()) + res["residual"] == pytest.approx(
        res["gap"], abs=1e-12
    )
    assert sum(res["r2_shapley"].values()) == pytest.approx(
        res["r2_full"] - res["r2_domain_only"], abs=1e-12
    )


def test_shapley_is_order_invariant() -> None:
    y, groups, domain = _toy()
    base = st.decompose(y, groups, domain)
    for perm in itertools.permutations(groups):
        res = st.decompose(y, {k: groups[k] for k in perm}, domain)
        for k in groups:
            assert res["contribution"][k] == pytest.approx(base["contribution"][k], abs=1e-12)
            assert res["r2_shapley"][k] == pytest.approx(base["r2_shapley"][k], abs=1e-12)


def test_gap_identity_on_noisy_random_data() -> None:
    """explained(S) = gap + gamma(S) for every subset (OLS with intercept and D3), and the Shapley
    contributions plus the residual reproduce the gap exactly on correlated, noisy data."""
    rng = np.random.default_rng(0)
    n = 300
    domain = (np.arange(n) >= 200).astype(float)
    a = rng.normal(size=(n, 2)) + domain[:, None]
    b = a[:, :1] * 0.8 + rng.normal(size=(n, 1)) - domain[:, None]
    c = rng.normal(size=(n, 3))
    y = a @ np.array([1.0, -2.0]) + 3 * b[:, 0] + c[:, 0] - 4 * domain + rng.normal(size=n)
    groups = {"a": a, "b": b, "c": c}
    fits = st.all_subset_fits(y, groups, domain)
    gap = y[domain == 0].mean() - y[domain == 1].mean()
    for fit in fits.values():
        assert fit["explained"] == pytest.approx(gap + fit["gamma"], abs=1e-9)
    res = st.decompose(y, groups, domain)
    assert sum(res["contribution"].values()) + res["residual"] == pytest.approx(gap, abs=1e-9)
    # lstsq coefficient on D3 agrees with statsmodels (HC3 table uses statsmodels on the design)
    feature_names = {"a": ["a1", "a2"], "b": ["b1"], "c": ["c1", "c2", "c3"]}
    hc3 = st.hc3_table(y, groups, feature_names, domain)
    assert hc3["coef_per_sd"]["domain_e3"] == pytest.approx(
        fits[frozenset(groups)]["gamma"], abs=1e-9
    )


def test_bootstrap_is_deterministic_under_seed_and_stratified() -> None:
    y, groups, domain = _toy()
    a = st.bootstrap_decompose(y, groups, domain, n_boot=15, seed=1234)
    b = st.bootstrap_decompose(y, groups, domain, n_boot=15, seed=1234)
    assert a == b
    rng = np.random.default_rng(1)
    idx = st.stratified_resample(domain, rng)
    assert int((domain[idx] == 0).sum()) == 8 and int((domain[idx] == 1).sum()) == 8
    noisy = y + np.random.default_rng(3).normal(size=len(y))
    c = st.bootstrap_decompose(noisy, groups, domain, n_boot=15, seed=1234)
    d = st.bootstrap_decompose(noisy, groups, domain, n_boot=15, seed=99)
    assert c["gap_ci"] != d["gap_ci"]


def test_vif_flags_collinearity_and_independence() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    vif = st.vif_table(
        {"x": x, "x_dup": 2 * x + 1e-9 * rng.normal(size=200), "z": rng.normal(size=200)}
    )
    assert vif["x"] > 1e6 and vif["z"] < 1.2
    assert st.vif_table({"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": np.array([1.0, 2.0, 3.0, 4.0])})[
        "a"
    ] == float("inf")


def test_assign_bands_thresholds_and_ties() -> None:
    counts = [1_000_000 - i for i in range(11_000)] + [5, 9, 10, 0, 0]
    tie = list(range(len(counts)))
    bands = g.assign_bands(counts, tie)
    assert (bands[:1000] == 2).all()  # rank 0..999 -> top1k
    assert (bands[1000:10000] == 3).all()  # rank 1000..9999 -> mid
    assert (bands[10000:11000] == 4).all()  # rank >= 10000 with count >= 10 -> tail_ge10
    assert bands[11000] == 1 and bands[11001] == 1  # count 5 and 9 -> rare
    assert bands[11002] == 4  # count 10 is NOT rare (rare is 1..9) and ranks past 10000
    assert bands[11003] == 0 and bands[11004] == 0  # unseen
    # ties are broken by the tiebreak key: with equal counts the smaller key ranks first
    eq = g.assign_bands([50] * 1200, list(range(1200)))
    assert (eq[:1000] == 2).all() and (eq[1000:] == 3).all()
    rev = g.assign_bands([50] * 1200, list(range(1199, -1, -1)))
    assert (rev[200:] == 2).all() and (rev[:200] == 3).all()


def test_word_core_and_function_word_mapping() -> None:
    ws = frozenset({"the", "of", "he", "it", "is"})
    assert g.word_core("▁Don't,") == "don't"
    assert g.word_core("▁...") == ""
    assert g.is_function_word("the", ws)
    assert g.is_function_word("don't", ws)  # don + t are clitic pieces
    assert g.is_function_word("it's", ws)  # it in list, s clitic
    assert not g.is_function_word("house", ws)
    assert not g.is_function_word("", ws)
    assert not g.is_function_word("l'ecole", ws)


def test_function_content_punct_token_categories_with_real_tokenizer() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(REPO_ROOT / "tokenizer" / "spm.model"))
    tok = g.Tok(sp)
    counts = np.zeros(sp.get_piece_size(), dtype=np.int64)
    ctx = g.FeatureContext(tok, counts, {"the": 100, "house": 3}, frozenset({"the", "of"}))
    out = g.sentence_features("La maison.", "The house of stone.", ctx)
    pieces = [tok.pieces[i] for i in out["ref_ids"]]
    cats = out["tok_cat"]
    assert cats[0] == 0 and pieces[0].startswith("▁")  # "The" -> function
    assert cats[-1] == 2  # final "." is punctuation
    assert 1 in cats  # "house"/"stone" -> content
    f = out["feats"]
    assert 0 < f["func_word_share"] < 1 and f["punct_token_share"] > 0
    # 'house' is rare (count 3), 'stone' absent from the training table -> unseen
    assert f["word_share_rare"] == pytest.approx(1 / 4) and f["word_share_unseen"] > 0
    assert f["tok_share_unseen"] == 1.0 - f["tok_share_top1k"] - f["tok_share_rare"]  # no mid band


def test_ratio_thresholds() -> None:
    f = g.ratio_flags
    assert f(1, 2)["moderate"] == 0.0  # r = 0.5 is not < 0.5
    assert f(49, 100)["moderate"] == 1.0 and f(49, 100)["severe"] == 0.0
    assert f(2, 1)["moderate"] == 0.0 and f(201, 100)["moderate"] == 1.0  # 2.0 not > 2.0
    assert f(33, 100)["severe"] == 0.0 and f(32, 100)["severe"] == 1.0  # 0.33 not < 0.33
    assert f(3, 1)["severe"] == 0.0 and f(301, 100)["severe"] == 1.0
    assert f(5, 0)["severe"] == 1.0  # empty reference
    assert f(10, 10)["moderate"] == 0.0


def test_register_features() -> None:
    r = g.register_features('-- Je suis ici, dit-il. "Nous"', 'I said "we" and he left')
    assert r["src_dialogue_start"] == 1.0
    assert r["src_quote_density"] == pytest.approx(2 / len('-- Je suis ici, dit-il. "Nous"'))
    assert r["ref_first_person_rate"] == pytest.approx(2 / 6)  # i, we of 6 words
    assert r["ref_pronoun_rate"] == pytest.approx(3 / 6)  # i, we, he
    assert r["src_first_person_rate"] == pytest.approx(2 / 6)  # je, nous of 6 words
    assert g.register_features("Il vient.", "He comes.")["src_dialogue_start"] == 0.0


def test_word_spans_and_word_list_file(tmp_path: Path) -> None:
    assert g.word_spans([True, False, True, True, False]) == [(0, 2), (2, 3), (3, 5)]
    assert g.word_spans([False, False]) == [(0, 2)]  # token 0 always opens a word
    p = tmp_path / "w.txt"
    p.write_bytes(b"# comment\nthe\n\nof\n")
    assert g.load_word_list(p) == frozenset({"the", "of"})
    real = g.load_word_list()
    assert len(real) == 318 and "the" in real and "house" not in real


def test_count_train_side_on_tiny_shard(tmp_path: Path) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(REPO_ROOT / "tokenizer" / "spm.model"))
    tok = g.Tok(sp)
    sents = ["The house.", "The old house stands."]
    ids = [np.array(sp.encode(s, out_type=int), dtype=np.uint16) for s in sents]
    off = np.array([0, len(ids[0]), len(ids[0]) + len(ids[1])], dtype=np.int64)
    np.savez(tmp_path / "shard_00000.npz", tgt=np.concatenate(ids), tgt_off=off)
    res = g.count_train_side([tmp_path / "shard_00000.npz"], "tgt", tok, want_words=True)
    assert res["n_sent"] == 2 and res["n_words"] == 2 + 4
    assert res["n_tokens"] == len(ids[0]) + len(ids[1])
    assert res["word_core_counts"]["the"] == 2 and res["word_core_counts"]["house"] == 2
    assert int(res["tok_counts"].sum()) == res["n_tokens"]


def test_mix_rate_breakdown_sums_to_gap() -> None:
    domain = np.array([0.0, 0.0, 1.0, 1.0])
    counts = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 4.0], [0.0, 5.0]])
    sums = np.array([[3.0, 4.0], [2.0, 6.0], [2.0, 20.0], [0.0, 30.0]])
    r = g.mix_rate_breakdown(sums, counts, domain, ["a", "b"])
    pooled1, pooled3 = (3 + 4 + 2 + 6) / 8, (2 + 20 + 30) / 10
    assert r["gap_e3_minus_e1"] == pytest.approx(pooled3 - pooled1)
    assert r["check_sum_terms"] == pytest.approx(r["gap_e3_minus_e1"], abs=1e-12)


def test_teacher_forced_nll_matches_manual_and_is_padding_invariant() -> None:
    import torch
    import torch.nn.functional as fn

    from nmt.hub import NMTModel

    torch.manual_seed(0)
    model = NMTModel(
        vocab_size=50, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32, max_len=64
    ).eval()
    srcs = [[5, 6, 7, 8, 9, 10], [11, 12], [13, 14, 15]]
    refs = [[20, 21, 22], [23, 24, 25, 26, 27], [28]]
    batched = g.teacher_forced_nll(model, srcs, refs, batch_size=3)
    solo = [
        g.teacher_forced_nll(model, [s], [r], batch_size=1)[0]
        for s, r in zip(srcs, refs, strict=True)
    ]
    for (nb, eb), (ns, es) in zip(batched, solo, strict=True):
        assert np.allclose(nb, ns, atol=1e-5) and eb == pytest.approx(es, abs=1e-5)
    cfg = model.config
    with torch.no_grad():
        logits = model(
            torch.tensor([srcs[0] + [cfg.eos_id]]), torch.tensor([[cfg.bos_id] + refs[0]])
        )
        logp = fn.log_softmax(logits.float(), -1)[0]
    manual = [-float(logp[i, t]) for i, t in enumerate(refs[0] + [cfg.eos_id])]
    assert np.allclose(batched[0][0], manual[:-1], atol=1e-5)
    assert batched[0][1] == pytest.approx(manual[-1], abs=1e-5)
    assert all(len(b[0]) == len(r) for b, r in zip(batched, refs, strict=True))


def test_write_json_uses_lf(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    g.write_json(p, {"label": g.LABEL, "a": [1, 2]})
    raw = p.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n") and b"EXPLORATORY / POST-HOC" in raw
