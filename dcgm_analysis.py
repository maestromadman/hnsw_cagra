#!/usr/bin/env python3
import csv
import math
import os
import sys

DCGM_FILE = "results/dcgm_raw.csv"
OUTPUT_FILE = "results/dcgm_summary.csv"


def parse_file(filepath):
    markers = []   # list of (line_idx, label)
    samples = []   # list of (line_idx, {field: float})
    col_order = []

    with open(filepath) as f:
        for line_idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# MARKER:"):
                rest = line[len("# MARKER:"):].strip()
                label = rest.split(" | ", 1)[0].strip()
                markers.append((line_idx, label))
            elif line.startswith("#Entity") or line.startswith("# Entity"):
                col_order = line.lstrip("#").split()  # ["Entity", "GPUTL", ...]
            elif line.startswith("GPU"):
                if not col_order:
                    continue
                parts = line.split()
                # ["GPU", "0", val0, val1, ...]
                vals = parts[2:]
                field_names = col_order[1:]  # skip "Entity"
                row = {}
                for i, name in enumerate(field_names):
                    if i < len(vals):
                        try:
                            row[name] = float(vals[i])
                        except ValueError:
                            pass
                if row:
                    samples.append((line_idx, row))

    return markers, samples


def window_samples(markers, samples):
    """Return (label, [rows]) for each inter-marker window."""
    windows = []
    for i, (m_idx, label) in enumerate(markers):
        next_idx = markers[i + 1][0] if i + 1 < len(markers) else math.inf
        rows = [row for (s_idx, row) in samples if m_idx < s_idx < next_idx]
        windows.append((label, rows))
    return windows


def compute_stats(rows):
    if not rows:
        return None
    stats = {}
    for field in ("GPUTL", "MCUTL", "FBUSD", "POWER", "SMCLK"):
        vals = [r[field] for r in rows if field in r]
        if vals:
            stats[f"mean_{field}"] = sum(vals) / len(vals)
            stats[f"peak_{field}"] = max(vals)
        else:
            stats[f"mean_{field}"] = float("nan")
            stats[f"peak_{field}"] = float("nan")
    return stats


def classify_bottleneck(mean_gputl, mean_mcutl, mean_power, mean_smclk):
    if mean_gputl < 5:
        return "idle"
    if mean_mcutl > mean_gputl + 20:
        return "memory-bandwidth bound"
    if mean_smclk < 1500 and mean_power > 55:
        return "power-throttled"
    if mean_gputl > 80 and mean_mcutl < mean_gputl - 20:
        return "compute-bound"
    return "balanced"


def parse_label(label):
    """Return (experiment, param_value, index_name) or None for START/unknown labels."""
    parts = label.split("|")
    if len(parts) == 3:
        exp_name, param_eq, index_name = parts
        vary_param = exp_name[len("exp_"):] if exp_name.startswith("exp_") else exp_name
        param_value = param_eq.split("=", 1)[1] if "=" in param_eq else param_eq
        return vary_param, param_value, index_name
    return None


def main():
    if not os.path.exists(DCGM_FILE):
        print(f"Error: {DCGM_FILE} not found", file=sys.stderr)
        sys.exit(1)

    markers, samples = parse_file(DCGM_FILE)
    if not markers:
        print("No markers found in DCGM file", file=sys.stderr)
        sys.exit(1)

    windows = window_samples(markers, samples)

    table_rows = []
    for label, rows in windows:
        parsed = parse_label(label)
        if parsed is None:
            continue  # START or unrecognised marker window

        vary_param, param_value, index_name = parsed
        stats = compute_stats(rows)
        if stats is None:
            continue

        bottleneck = classify_bottleneck(
            stats["mean_GPUTL"], stats["mean_MCUTL"],
            stats["mean_POWER"], stats["mean_SMCLK"],
        )

        table_rows.append({
            "experiment":  vary_param,
            "param_value": param_value,
            "index":       index_name,
            "mean_GPUTL":  f"{stats['mean_GPUTL']:.1f}",
            "mean_MCUTL":  f"{stats['mean_MCUTL']:.1f}",
            "mean_POWER":  f"{stats['mean_POWER']:.1f}",
            "peak_SMCLK":  f"{stats['peak_SMCLK']:.0f}",
            "bottleneck":  bottleneck,
        })

    col_headers = ["experiment", "param_value", "index",
                   "mean_GPUTL", "mean_MCUTL", "mean_POWER", "peak_SMCLK", "bottleneck"]

    if not table_rows:
        print("No data rows to display.")
        return

    col_widths = {
        h: max(len(h), max(len(str(r[h])) for r in table_rows))
        for h in col_headers
    }

    sep = "─" * (sum(col_widths.values()) + 3 * (len(col_headers) - 1))
    header_line = "   ".join(h.ljust(col_widths[h]) for h in col_headers)
    print(sep)
    print(header_line)
    print(sep)
    for r in table_rows:
        print("   ".join(str(r[h]).ljust(col_widths[h]) for h in col_headers))
    print(sep)

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_headers)
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
