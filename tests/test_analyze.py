"""Tests for analyze.py (aggregation, histogram, paper comparison)."""
import csv
import json

from analyze import PAPER_TABLE_10, load_results, main, summarize


def _write_csv(path, values):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_id", "labels", "states", "n_sb", "n_sr", "n_mc", "n_arx", "min_active"])
        for k, v in enumerate(values):
            w.writerow([k, "0" * 10, "0123" * 5, 5, 5, 5, 5, v])
    path.with_suffix(".meta.json").write_text(json.dumps({"initial_state": 0, "arx_mode": "identity"}))


def test_paper_values_are_quoted_for_comparison_only():
    assert PAPER_TABLE_10 == {"min": 36, "mean": 87.17, "median": 85, "max": 133}


def test_summarize_basic_statistics(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [11, 25, 26, 14])
    s = summarize(load_results(p))
    assert (s["min"], s["max"], s["mean"], s["median"], s["count"]) == (11, 26, 19.0, 19.5, 4)


def test_main_writes_summary_and_histogram(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [11, 25, 26, 14, 11])
    out_md = tmp_path / "summary.md"
    out_png = tmp_path / "hist.png"
    main(["--csv", str(p), "--summary", str(out_md), "--histogram", str(out_png)])
    text = out_md.read_text()
    assert "87.17" in text and "| 11 |" in text and "26" in text
    assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_includes_variant_tables(tmp_path):
    p = tmp_path / "a.csv"
    v = tmp_path / "variant.csv"
    _write_csv(p, [11, 25])
    _write_csv(v, [0, 3, 3])
    out_md = tmp_path / "summary.md"
    main(["--csv", str(p), "--variants", str(v), "--summary", str(out_md), "--histogram", str(tmp_path / "h.png")])
    assert "variant" in out_md.read_text()
