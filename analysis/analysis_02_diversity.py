"""
Bước 2 — Đo độ đa dạng và quét toàn bộ tổ hợp ensemble 3 model.

Chỉ cần CPU, chạy vài phút. Đầu vào là các file preds_*.npz từ bước 1.

Câu hỏi script này trả lời:
  1. Hai model có accuracy gần như nhau thì sai trên cùng những ảnh nào?
  2. Trong 816 tổ hợp 3 model, lợi ích của ensemble đi theo độ chính xác
     trung bình của các thành viên hay đi theo độ bất đồng giữa chúng?
  3. Chọn 3 model đa dạng nhất có thắng 3 model chính xác nhất không?

Đầu ra (thư mục out/):
  pairwise.csv          153 cặp: bất đồng, Q-statistic, kappa, double-fault, McNemar
  triples.csv           816 tổ hợp: accuracy ensemble, lợi ích, các chỉ số đa dạng
  strategies.csv        so sánh các cách chọn thành viên
  regression.txt        lợi ích ~ accuracy trung bình + độ bất đồng
  summary.txt           tóm tắt bằng lời
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default="results/seed36", help="thư mục chứa preds_*.npz")
_ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định: <dir>/../out_<tên>)")
_args, _ = _ap.parse_known_args()

PRED_DIR = Path(_args.dir)
OUT_DIR = Path(_args.out) if _args.out else PRED_DIR.parent / f"out_{PRED_DIR.name}"
RNG = np.random.default_rng(36)
N_BOOT = 10_000


# ------------------------------------------------------------------ nạp


def load_all():
    preds, probs, models = {}, {}, []
    y_true = None

    for f in sorted(PRED_DIR.glob("preds_*.npz")):
        name = f.stem.replace("preds_", "")
        d = np.load(f)
        if y_true is None:
            y_true = d["y_true"].astype(int)
        elif not np.array_equal(y_true, d["y_true"].astype(int)):
            raise SystemExit(
                f"{name}: nhãn thật khác với các file trước.\n"
                "Nghĩa là thứ tự ảnh không thống nhất — chạy lại bước 1."
            )
        preds[name] = d["y_pred"].astype(int)
        probs[name] = d["y_prob"].astype(np.float64)
        models.append(name)

    if len(models) < 3:
        raise SystemExit(f"Chỉ có {len(models)} model, cần tối thiểu 3.")

    return models, preds, probs, y_true


def split_name(name):
    """'dense_net_GAP_BN_FC_Drop' -> ('dense_net', 'GAP_BN_FC_Drop')"""
    for head in ("SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"):
        if name.endswith("_" + head):
            return name[: -len(head) - 1], head
    return name, "?"


# ------------------------------------------------------------- từng cặp


def mcnemar(correct_a, correct_b):
    """McNemar trên hai vector đúng/sai. Dưới 25 cặp lệch thì dùng nhị thức
    chính xác, còn lại dùng chi-square có hiệu chỉnh liên tục."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(stats.chi2.sf(chi2, df=1))
    return b, c, p


def pairwise_table(models, preds, y_true):
    correct = {m: preds[m] == y_true for m in models}
    rows = []

    for a, b in itertools.combinations(models, 2):
        ca, cb = correct[a], correct[b]
        n11 = int(np.sum(ca & cb))
        n10 = int(np.sum(ca & ~cb))
        n01 = int(np.sum(~ca & cb))
        n00 = int(np.sum(~ca & ~cb))
        n = n11 + n10 + n01 + n00

        denom = n11 * n00 + n01 * n10
        q = (n11 * n00 - n01 * n10) / denom if denom else np.nan

        _, _, p = mcnemar(ca, cb)

        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "backbone_a": split_name(a)[0],
                "backbone_b": split_name(b)[0],
                "same_backbone": split_name(a)[0] == split_name(b)[0],
                "same_head": split_name(a)[1] == split_name(b)[1],
                # bất đồng trên nhãn dự đoán: bao nhiêu ảnh hai model đoán khác nhau
                "disagree_pred": float(np.mean(preds[a] != preds[b])),
                "n_disagree_pred": int(np.sum(preds[a] != preds[b])),
                # bất đồng trên đúng/sai: một đúng một sai
                "disagree_correct": (n10 + n01) / n,
                "q_statistic": q,
                "kappa_pred": float(cohen_kappa_score(preds[a], preds[b])),
                "double_fault": n00 / n,
                "both_correct": n11 / n,
                "mcnemar_p": p,
                "acc_a": float(ca.mean() * 100),
                "acc_b": float(cb.mean() * 100),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------- tổ hợp bộ ba


def triples_table(models, preds, probs, y_true, pair_df):
    # tra cứu nhanh chỉ số từng cặp
    pair_key = {}
    for r in pair_df.itertuples():
        pair_key[frozenset((r.model_a, r.model_b))] = r

    acc_single = {m: float((preds[m] == y_true).mean() * 100) for m in models}
    rows = []

    for combo in itertools.combinations(models, 3):
        p_mean = sum(probs[m] for m in combo) / 3.0
        y_ens = p_mean.argmax(axis=1)

        pairs = [pair_key[frozenset(c)] for c in itertools.combinations(combo, 2)]
        member_acc = [acc_single[m] for m in combo]
        backbones = {split_name(m)[0] for m in combo}
        heads = {split_name(m)[1] for m in combo}

        rows.append(
            {
                "models": " + ".join(combo),
                "ens_acc": float((y_ens == y_true).mean() * 100),
                "ens_macro_f1": float(f1_score(y_true, y_ens, average="macro") * 100),
                "mean_member_acc": float(np.mean(member_acc)),
                "best_member_acc": float(np.max(member_acc)),
                "worst_member_acc": float(np.min(member_acc)),
                "n_distinct_backbones": len(backbones),
                "n_distinct_heads": len(heads),
                "mean_disagree_pred": float(np.mean([p.disagree_pred for p in pairs])),
                "mean_q": float(np.mean([p.q_statistic for p in pairs])),
                "mean_kappa": float(np.mean([p.kappa_pred for p in pairs])),
                "mean_double_fault": float(np.mean([p.double_fault for p in pairs])),
            }
        )

    df = pd.DataFrame(rows)
    df["gain_over_best"] = df.ens_acc - df.best_member_acc
    df["gain_over_mean"] = df.ens_acc - df.mean_member_acc
    return df.sort_values("ens_acc", ascending=False).reset_index(drop=True)


# ------------------------------------------------------- cách chọn thành viên


def strategies_table(models, preds, probs, y_true, triples, pair_df):
    acc = {m: float((preds[m] == y_true).mean() * 100) for m in models}
    by_acc = sorted(models, key=lambda m: -acc[m])

    def ens_acc(combo):
        p = sum(probs[m] for m in combo) / len(combo)
        return float((p.argmax(axis=1) == y_true).mean() * 100)

    def mean_disagree(combo):
        sub = pair_df[
            pair_df.apply(
                lambda r: r.model_a in combo and r.model_b in combo, axis=1
            )
        ]
        return float(sub.disagree_pred.mean())

    rows = []

    # 1. ba model chính xác nhất
    top3 = tuple(by_acc[:3])
    rows.append(("Top-3 theo accuracy", top3))

    # 2. ba model chính xác nhất nhưng khác backbone
    seen, pick = set(), []
    for m in by_acc:
        bb = split_name(m)[0]
        if bb not in seen:
            seen.add(bb)
            pick.append(m)
        if len(pick) == 3:
            break
    rows.append(("Top-3, mỗi backbone một model", tuple(pick)))

    # 3. bộ ba bất đồng nhiều nhất, bất kể accuracy
    most_div = triples.sort_values("mean_disagree_pred", ascending=False).iloc[0]
    rows.append(("Bất đồng cao nhất", tuple(most_div.models.split(" + "))))

    # 4. bất đồng cao nhất trong nhóm khá: thành viên kém nhất cách model tốt
    #    nhất không quá 2 điểm. Đây là cách chọn dùng được trong thực tế.
    best = max(acc.values())
    good = triples[triples.worst_member_acc >= best - 2.0]
    if len(good):
        g = good.sort_values("mean_disagree_pred", ascending=False).iloc[0]
        rows.append(("Bất đồng cao nhất trong nhóm khá (±2 điểm)", tuple(g.models.split(" + "))))

    # 5. trần trên: tổ hợp tốt nhất nếu biết trước đáp án
    rows.append(("Tổ hợp tốt nhất (trần trên)", tuple(triples.iloc[0].models.split(" + "))))

    out = []
    for label, combo in rows:
        out.append(
            {
                "strategy": label,
                "models": " + ".join(combo),
                "ens_acc": ens_acc(combo),
                "best_member_acc": max(acc[m] for m in combo),
                "gain_over_best": ens_acc(combo) - max(acc[m] for m in combo),
                "mean_disagree_pred": mean_disagree(combo),
            }
        )

    # 6. bộ ba ngẫu nhiên, lấy trung bình để làm mốc
    sample = triples.sample(n=min(200, len(triples)), random_state=36)
    out.append(
        {
            "strategy": "Ngẫu nhiên (trung bình 200 bộ)",
            "models": "-",
            "ens_acc": float(sample.ens_acc.mean()),
            "best_member_acc": float(sample.best_member_acc.mean()),
            "gain_over_best": float(sample.gain_over_best.mean()),
            "mean_disagree_pred": float(sample.mean_disagree_pred.mean()),
        }
    )

    return pd.DataFrame(out)


# ------------------------------------------------------------- hồi quy


def regression_report(triples):
    """Lợi ích ensemble ~ accuracy trung bình + độ bất đồng.

    Nếu hệ số của độ bất đồng vẫn có ý nghĩa sau khi đã kiểm soát accuracy
    trung bình, thì độ đa dạng mang thông tin mà accuracy không có. Đó là
    toàn bộ luận điểm của bài.
    """
    y = triples.gain_over_best.values
    x1 = triples.mean_member_acc.values
    x2 = triples.mean_disagree_pred.values

    def fit(cols, names):
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        sigma2 = resid @ resid / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
        return beta, se, t, p, r2, names

    lines = []
    for cols, names in (
        ([x1], ["accuracy trung bình"]),
        ([x2], ["độ bất đồng"]),
        ([x1, x2], ["accuracy trung bình", "độ bất đồng"]),
    ):
        beta, se, t, p, r2, nm = fit(cols, names)
        lines.append(f"Mô hình: lợi ích ~ {' + '.join(nm)}")
        lines.append(f"  R² = {r2:.4f}")
        for i, name in enumerate(["hằng số"] + nm):
            lines.append(
                f"  {name:<22} hệ số {beta[i]:+.4f}  SE {se[i]:.4f}  "
                f"t {t[i]:+.2f}  p {p[i]:.3g}"
            )
        lines.append("")

    rho_acc = stats.spearmanr(triples.mean_member_acc, triples.gain_over_best)
    rho_div = stats.spearmanr(triples.mean_disagree_pred, triples.gain_over_best)
    lines.append("Tương quan hạng Spearman với lợi ích so với thành viên tốt nhất:")
    lines.append(f"  accuracy trung bình  rho {rho_acc.statistic:+.3f}  p {rho_acc.pvalue:.3g}")
    lines.append(f"  độ bất đồng          rho {rho_div.statistic:+.3f}  p {rho_div.pvalue:.3g}")

    return "\n".join(lines)


def bootstrap_gap(y_true, pred_a, pred_b, n_boot=N_BOOT):
    """KTC 95% cho chênh lệch accuracy, bootstrap ghép cặp theo ảnh."""
    ca = (pred_a == y_true).astype(float)
    cb = (pred_b == y_true).astype(float)
    n = len(y_true)
    idx = RNG.integers(0, n, size=(n_boot, n))
    diffs = ca[idx].mean(axis=1) - cb[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5) * 100), float(np.percentile(diffs, 97.5) * 100)


# -------------------------------------------------------------------- main


def main():
    OUT_DIR.mkdir(exist_ok=True)
    models, preds, probs, y_true = load_all()
    print(f"Nạp {len(models)} model, {len(y_true)} ảnh test\n")

    pair_df = pairwise_table(models, preds, y_true)
    pair_df.to_csv(OUT_DIR / "pairwise.csv", index=False)

    triples = triples_table(models, preds, probs, y_true, pair_df)
    triples.to_csv(OUT_DIR / "triples.csv", index=False)

    strat = strategies_table(models, preds, probs, y_true, triples, pair_df)
    strat.to_csv(OUT_DIR / "strategies.csv", index=False)

    reg = regression_report(triples)
    (OUT_DIR / "regression.txt").write_text(reg, encoding="utf-8")

    # ---- tóm tắt
    acc = {m: float((preds[m] == y_true).mean() * 100) for m in models}
    best_model = max(acc, key=acc.get)
    best_tri = triples.iloc[0]
    best_combo = best_tri.models.split(" + ")
    p_ens = sum(probs[m] for m in best_combo) / 3.0
    lo, hi = bootstrap_gap(y_true, p_ens.argmax(axis=1), preds[best_model])

    indist = pair_df[pair_df.mcnemar_p > 0.05]

    S = []
    S.append(f"Model đơn tốt nhất: {best_model} — {acc[best_model]:.2f}%")
    S.append(f"Ensemble tốt nhất:  {best_tri.models} — {best_tri.ens_acc:.2f}%")
    S.append(f"Chênh lệch: {best_tri.ens_acc - acc[best_model]:+.2f} điểm, KTC 95% [{lo:+.2f}, {hi:+.2f}]")
    S.append("")
    S.append(f"Số cặp không phân biệt được bằng McNemar (p > 0,05): {len(indist)}/{len(pair_df)}")
    if len(indist):
        S.append(
            f"  Dù vậy các cặp đó vẫn đoán khác nhau trên trung bình "
            f"{indist.n_disagree_pred.mean():.0f} ảnh "
            f"({indist.n_disagree_pred.min()}–{indist.n_disagree_pred.max()})."
        )
        S.append("  Accuracy ngang nhau không có nghĩa là sai giống nhau.")
    S.append("")

    by_bb = triples.groupby("n_distinct_backbones").agg(
        so_to_hop=("ens_acc", "size"),
        acc_tb=("ens_acc", "mean"),
        acc_max=("ens_acc", "max"),
        loi_ich_tb=("gain_over_best", "mean"),
        bat_dong_tb=("mean_disagree_pred", "mean"),
    )
    S.append("Theo số backbone khác nhau trong bộ ba:")
    S.append(by_bb.round(3).to_string())
    S.append("")
    S.append("Các cách chọn thành viên:")
    S.append(strat.round(3).to_string(index=False))
    S.append("")
    S.append(reg)

    text = "\n".join(S)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
