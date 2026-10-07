"""Aggregate per-path minima (results/all_paths.csv) and compare with the paper.

Produces results/summary.md (tables) and results/histogram.png.  The paper
values in PAPER_TABLE_10 are quoted for comparison only; nothing in the
pipeline uses them as input.

Usage:
    python analyze.py
    python analyze.py --variants results/variants/*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fsm import describe  # noqa: E402

# Alawida et al., "AEX-128: A Key-Dependent Finite-State SPN-ARX Hybrid Block Cipher",
# IEEE Access 2026, doi:10.1109/ACCESS.2026.3730739, Table 10 (all 3^10 = 59,049 label sequences).
PAPER_TABLE_10 = {"min": 36, "mean": 87.17, "median": 85, "max": 133}
PAPER_CITATION = "Alawida et al., IEEE Access 2026, doi:10.1109/ACCESS.2026.3730739, Table 10"


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"labels": str, "states": str})
    meta_path = Path(csv_path).with_suffix(".meta.json")
    df.attrs["meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    df.attrs["source"] = Path(csv_path).as_posix()
    return df


def summarize(df: pd.DataFrame) -> dict:
    v = df["min_active"]
    counts = v.value_counts().sort_index()
    return {
        "count": int(len(v)),
        "min": int(v.min()),
        "mean": round(float(v.mean()), 2),
        "median": float(v.median()),
        "max": int(v.max()),
        "distribution": {int(k): int(c) for k, c in counts.items()},
        "argmin": df[v == v.min()].head(5).to_dict("records"),
        "argmax": df[v == v.max()].head(5).to_dict("records"),
        "n_argmin": int((v == v.min()).sum()),
        "n_argmax": int((v == v.max()).sum()),
        "by_n_sb": {int(k): (round(float(g["min_active"].mean()), 2), int(g["min_active"].min()), int(len(g)))
                    for k, g in df.groupby("n_sb")} if "n_sb" in df else {},
    }


def _fmt(x):
    return f"{x:g}" if isinstance(x, float) else str(x)


def comparison_table(s: dict) -> str:
    lines = ["| Metric | This reproduction | Paper Table 10 | Difference |", "|---|---|---|---|"]
    for key, label in (("min", "Minimum"), ("mean", "Average"), ("median", "Median"), ("max", "Maximum")):
        ours, paper = s[key], PAPER_TABLE_10[key]
        lines.append(f"| {label} | {_fmt(ours)} | {_fmt(paper)} | {_fmt(round(ours - paper, 2))} |")
    return "\n".join(lines)


def _path_table(rows) -> str:
    lines = ["| path id | labels | states | n_sb | min | operations |", "|---|---|---|---|---|---|"]
    for r in rows:
        ops = describe(tuple(int(ch) for ch in r["states"]))
        lines.append(f"| {r['path_id']} | {r['labels']} | {r['states']} | {r['n_sb']} | {r['min_active']} | {ops} |")
    return "\n".join(lines)


def _variant_row(name: str, df: pd.DataFrame) -> str:
    s, m = summarize(df), df.attrs.get("meta", {})
    return (f"| {name} | S{m.get('initial_state', '?')} | {m.get('arx_mode', '?')} | {s['count']} | "
            f"{s['min']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {s['max']} |")


def render_summary(df: pd.DataFrame, variants: list[tuple[str, pd.DataFrame]], histogram: Path) -> str:
    s, m = summarize(df), df.attrs.get("meta", {})
    out = [
        "# AEX-128 MILP active S-box reproduction: results",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{df.attrs.get('source')}`.",
        "",
        "## Primary run",
        "",
        f"- initial state: S{m.get('initial_state', '?')}; ARX model: {m.get('arx_mode', '?')}; "
        f"solver: {m.get('solver', '?')}; rows: {s['count']} of {m.get('total_paths', '?')} paths; "
        f"distinct MILPs solved: {m.get('unique_models', '?')}; wall time: {m.get('elapsed_seconds', '?')} s",
        "",
        f"Comparison with the paper ({PAPER_CITATION}):",
        "",
        comparison_table(s),
        "",
        "## Distribution of the per-path minimum",
        "",
        f"![histogram]({histogram.name})",
        "",
        "| min active S-boxes | paths | share |", "|---|---|---|",
    ]
    out += [f"| {k} | {c} | {100 * c / s['count']:.2f}% |" for k, c in s["distribution"].items()]
    out += ["", "## Minimum by number of SubBytes layers on the path", "",
            "| SubBytes layers | paths | mean min | smallest min |", "|---|---|---|---|"]
    out += [f"| {k} | {n} | {mean} | {mn} |" for k, (mean, mn, n) in s["by_n_sb"].items()]
    out += ["", f"## Paths attaining the minimum ({s['n_argmin']} paths, first 5)", "", _path_table(s["argmin"]),
            "", f"## Paths attaining the maximum ({s['n_argmax']} paths, first 5)", "", _path_table(s["argmax"])]
    if variants:
        out += ["", "## Sensitivity variants", "",
                "| variant | initial | ARX | rows | min | mean | median | max |", "|---|---|---|---|---|---|---|---|",
                _variant_row("primary", df)]
        out += [_variant_row(name, vdf) for name, vdf in variants]
    return "\n".join(out) + "\n"


def write_histogram(df: pd.DataFrame, path: Path) -> None:
    counts = df["min_active"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts.index, counts.values, width=0.8, color="#3b6ea5")
    for key, style in (("min", ":"), ("median", "--"), ("max", ":")):
        ax.axvline(PAPER_TABLE_10[key], color="#b03a2e", linestyle=style, linewidth=1,
                   label=f"paper {key} = {PAPER_TABLE_10[key]}")
    ax.set_xlabel("minimum number of active S-boxes on the path")
    ax.set_ylabel("number of label sequences")
    ax.set_title(f"AEX-128, {len(df)} FSM paths, truncated MILP (ARX = {df.attrs.get('meta', {}).get('arx_mode', '?')})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("results") / "all_paths.csv")
    parser.add_argument("--variants", type=Path, nargs="*", default=[])
    parser.add_argument("--summary", type=Path, default=Path("results") / "summary.md")
    parser.add_argument("--histogram", type=Path, default=Path("results") / "histogram.png")
    args = parser.parse_args(argv)

    df = load_results(args.csv)
    variants = [(p.stem, load_results(p)) for p in args.variants if Path(p).exists()]
    write_histogram(df, args.histogram)
    text = render_summary(df, variants, args.histogram)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(text, encoding="utf-8")
    print(comparison_table(summarize(df)))
    print(f"wrote {args.summary} and {args.histogram}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
