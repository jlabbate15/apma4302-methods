#!/usr/bin/env python3
import os
import re
import csv

# Must match p8.zsh
BASE_GRID = 9
PROCS = [1, 2, 4]
REFINES = [2, 3, 4, 5, 6]
RUNS_ROOT = "runs"

rel_err_pat = re.compile(r"\|u-u_exact\|_2\s*/\s*\|u_exact\|_2\s*=\s*([0-9eE+\-.]+)")
snes_mon_pat = re.compile(r"^\s*\d+\s+SNES Function norm\b", re.MULTILINE)

# Runtime extraction from -log_view is less standardized; these are best-effort.
runtime_pats = [
    re.compile(r"Total\s+time\s*[:=]\s*([0-9eE+\-.]+)\s*(s|sec|seconds)?", re.IGNORECASE),
    re.compile(r"Time \(sec\)\s*[:=]\s*([0-9eE+\-.]+)", re.IGNORECASE),
    re.compile(r"Total\s+runtime\s*[:=]\s*([0-9eE+\-.]+)\s*(s|sec|seconds)?", re.IGNORECASE),
]

def parse_runtime(text: str):
    for pat in runtime_pats:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None

def parse_snes_iters(text: str):
    # With -snes_monitor, PETSc prints one line per SNES iteration:
    matches = list(snes_mon_pat.finditer(text))
    if matches:
        # Often iterations start at 0, so count = occurrences is fine.
        return len(matches)

    # Fallback if monitors aren't present for some reason
    m = re.search(r"converged.*?in\s+(\d+)\s+iterations", text, re.IGNORECASE | re.DOTALL)
    if m:
        return int(m.group(1))
    return None

def parse_rel_err(text: str):
    m = rel_err_pat.search(text)
    if not m:
        return None
    return float(m.group(1))

rows = []
for nprocs in PROCS:
    for ref in REFINES:
        log_path = os.path.join(RUNS_ROOT, f"p8_n{nprocs}_ref{ref}", "run.log")
        if not os.path.exists(log_path):
            continue

        with open(log_path, "r", errors="ignore") as f:
            text = f.read()

        grid = (BASE_GRID - 1) * (2 ** ref) + 1
        rows.append({
            "nprocs": nprocs,
            "refine": ref,
            "da_grid": grid,
            "runtime_sec": parse_runtime(text),
            "snes_iters": parse_snes_iters(text),
            "relative_error_L2": parse_rel_err(text),
        })

out_csv = os.path.join(RUNS_ROOT, "p8_summary.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
        "nprocs","refine","da_grid","runtime_sec","snes_iters","relative_error_L2"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {out_csv} with {len(rows)} rows.")