# petsc_scaling_from_logs.py
#
# Parses PETSc ASCII -log_view output from files named like:
#   <prefix>_N257_P064_p08.o.*
# where:
#   N257 => global problem linear dimension n, so DoF = n^3
#   P064 => MPI ranks
#   p08  => ranks per node (optional meta)
#
# Produces strong/weak scaling plots for total time (if found) and for selected events.

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os



# ----------------------------
# Filename parsing
# ----------------------------

@dataclass(frozen=True)
class Meta:
    path: Path
    prefix: str
    n: int              # from N257
    dof: int            # n^3
    P: int              # from P064
    ppn: Optional[int]  # from p08 (processors per node), may be None


FNAME_RE = re.compile(
    r"""
    ^(?P<prefix>.+?)         # prefix up to first _N
    _N(?P<n>\d+)             # N257
    _P(?P<P>\d+)             # P064
    (?:_p(?P<ppn>\d+))?      # optional _p08
    \.o\..*$                 # .o.*
    """,
    re.VERBOSE,
)

def parse_meta(path: Path) -> Meta:
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match convention: {path.name}")

    prefix = m.group("prefix")
    n = int(m.group("n"))
    P = int(m.group("P"))
    ppn = int(m.group("ppn")) if m.group("ppn") is not None else None
    dof = n ** 3
    return Meta(path=path, prefix=prefix, n=n, dof=dof, P=P, ppn=ppn)


# ----------------------------
# PETSc log parsing (ASCII -log_view)
# ----------------------------

def parse_total_time(text: str) -> Optional[float]:
    # PETSc output varies; try several common patterns.
    pats = [
        r"Time\s*\(sec\)\s*:\s*([0-9.eE+-]+)",
        r"Total\s+Time\s*:\s*([0-9.eE+-]+)",
        r"elapsed\s+time\s*[:=]\s*([0-9.eE+-]+)",
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def parse_event_times(text: str) -> pd.DataFrame:
    """
    Extract a simple table: event -> time.
    Works for many PETSc ASCII -log_view variants where an "Event ... Time" table exists.
    """
    lines = text.splitlines()

    header_i = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Event") and ("Time" in line):
            header_i = i
            break
    if header_i is None:
        return pd.DataFrame(columns=["event", "time"])

    header = lines[header_i].split()
    time_idx = None
    for j, c in enumerate(header):
        if c.lower().startswith("time"):
            time_idx = j
            break
    if time_idx is None:
        return pd.DataFrame(columns=["event", "time"])

    data = []
    for line in lines[header_i + 1 :]:
        if not line.strip():
            break
        if set(line.strip()) <= set("-="):
            continue
        parts = line.split()
        if len(parts) <= time_idx:
            continue
        event = parts[0]
        try:
            t = float(parts[time_idx])
        except ValueError:
            continue
        data.append((event, t))

    return pd.DataFrame(data, columns=["event", "time"])


def parse_log_file(path: Path) -> dict[str, float]:
    text = path.read_text(errors="ignore")
    out: dict[str, float] = {}

    tt = parse_total_time(text)
    if tt is not None:
        out["__total__"] = tt

    ev = parse_event_times(text)
    for _, r in ev.iterrows():
        out[str(r["event"])] = float(r["time"])
    return out


# ----------------------------
# Loading to a tidy DataFrame
# ----------------------------

def load_logs(log_dir: str | Path, glob_pattern: str = "*.o.*") -> pd.DataFrame:
    log_dir = Path(log_dir)
    records = []

    for f in sorted(log_dir.glob(glob_pattern)):
        meta = parse_meta(f)
        metrics = parse_log_file(f)
        for metric, t in metrics.items():
            records.append(
                dict(
                    prefix=meta.prefix,
                    n=meta.n,
                    dof=meta.dof,
                    P=meta.P,
                    ppn=meta.ppn,
                    metric=metric,
                    time=t,
                    file=str(f),
                )
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"No logs parsed in {log_dir} with pattern {glob_pattern}")
    return df


# ----------------------------
# Scaling calculations
# ----------------------------

def strong_scaling(df: pd.DataFrame, metric: str, prefix: Optional[str] = None, n: Optional[int] = None) -> pd.DataFrame:
    sub = df[df["metric"] == metric].copy()
    if prefix is not None:
        sub = sub[sub["prefix"] == prefix]
    if n is not None:
        sub = sub[sub["n"] == n]

    sub = sub.sort_values("P")
    if sub.empty:
        raise ValueError("No data after filtering; check prefix/n/metric filters.")

    P0 = int(sub["P"].min())
    t0 = float(sub.loc[sub["P"] == P0, "time"].iloc[0])

    sub["speedup"] = t0 / sub["time"]
    sub["efficiency"] = sub["speedup"] / (sub["P"] / P0)
    return sub[["prefix", "n", "dof", "P", "ppn", "time", "speedup", "efficiency"]]


def weak_scaling(df: pd.DataFrame, metric: str, prefix: Optional[str] = None, dof_per_rank_tol: float = 1e-12) -> pd.DataFrame:
    """
    Weak scaling: expect dof/P ~ constant.
    We compute dof_per_rank and group by it (with rounding for stability).
    """
    sub = df[df["metric"] == metric].copy()
    if prefix is not None:
        sub = sub[sub["prefix"] == prefix]
    if sub.empty:
        raise ValueError("No data after filtering; check prefix/metric filters.")

    sub["dof_per_rank"] = sub["dof"] / sub["P"]

    # Grouping floating values: round to a reasonable number of significant digits.
    # You can also quantize by integer if dof_per_rank should be integer.
    sub["dof_per_rank_key"] = sub["dof_per_rank"].map(lambda x: float(f"{x:.6g}"))

    frames = []
    for key, g in sub.groupby("dof_per_rank_key"):
        g = g.sort_values("P").copy()
        P0 = int(g["P"].min())
        t0 = float(g.loc[g["P"] == P0, "time"].iloc[0])
        g["weak_eff"] = t0 / g["time"]
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    return out[["prefix", "n", "dof", "P", "ppn", "dof_per_rank", "time", "weak_eff"]].sort_values(["dof_per_rank", "P"])


# ----------------------------
# Plotting
# ----------------------------

def plot_strong(ss: pd.DataFrame, metric: str, out: Optional[str] = None):
    formatter = ticker.StrMethodFormatter('{x:.0f}')

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].plot(ss["P"], ss["time"], marker="o",markersize=8, markeredgecolor='c',linestyle='')
    ax[0].set_xscale("log", base=2)
    ax[0].xaxis.set_major_formatter(formatter)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("MPI ranks (P)")
    ax[0].set_ylabel("Time (s)")
    ax[0].grid()
    ax[0].set_title(f"Strong scaling runtime ({metric})")

    ax[1].plot(ss["P"], ss["speedup"], marker="o",markersize=8, markeredgecolor='c',linestyle='', label="Measured")
    P0 = int(ss["P"].min())
    ax[1].plot(ss["P"], ss["P"] / P0, linestyle="--", label="Ideal")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log", base=2)
    ax[1].xaxis.set_major_formatter(formatter)
    ax[1].yaxis.set_major_formatter(formatter)
    ax[1].set_xlabel("MPI ranks (P)")
    ax[1].set_ylabel("Speedup")
    ax[1].set_title("Speedup")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(ss["P"], ss["efficiency"], marker="o",markersize=8, markeredgecolor='c',linestyle='')
    ax[2].set_xscale("log", base=2)
    ax[2].xaxis.set_major_formatter(formatter)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlabel("MPI ranks (P)")
    ax[2].set_ylabel("Parallel efficiency")
    ax[2].grid()
    ax[2].set_title("Efficiency")

    title = f"{ss['prefix'].iloc[0]}  N={ss['n'].iloc[0]}  (DoF={ss['dof'].iloc[0]:,})"
    fig.suptitle(title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
    return fig


def plot_weak(ws: pd.DataFrame, metric: str, out: Optional[str] = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(ws["P"], ws["time"], marker="o")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("MPI ranks (P)")
    ax[0].set_ylabel("Time (s)")
    ax[0].set_title(f"Weak scaling runtime ({metric})")

    ax[1].plot(ws["P"], ws["weak_eff"], marker="o", label="Measured")
    ax[1].axhline(1.0, linestyle="--", color="k", linewidth=1, label="Ideal")
    ax[1].set_xscale("log", base=2)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("MPI ranks (P)")
    ax[1].set_ylabel("Weak scaling efficiency")
    ax[1].set_title("Efficiency")
    ax[1].legend()

    # If multiple dof_per_rank groups exist, title with the first group; otherwise just prefix.
    prefix = ws["prefix"].iloc[0]
    dpr = ws["dof_per_rank"].iloc[0]
    fig.suptitle(f"{prefix}  DoF/rank≈{dpr:,.3g}")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
    return fig


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    LOG_DIR = "logs"
    PATTERN = "*.o.*"  # matches your .o.* files
    parser = argparse.ArgumentParser(description="Process a directory.")
    parser.add_argument(
        "directory",
        help="Directory name/path to process"
    )
    
    args=parser.parse_args()

    directory = args.directory
    if not os.path.isdir(directory):
        parser.error(f"Not a valid directory: {directory}")

    #PATTERN=args.pattern

    # use `directory` below
    print(f"Using directory: {directory}")
    df = load_logs(directory, PATTERN)
    print(df.columns)

    # Choose the metric: "__total__" if total time is detected, else pick an event like "KSPSolve"
    metric = "__total__" if (df["metric"] == "__total__").any() else "KSPSolve"

    # Pick a prefix/n for strong scaling (fixed N). If you have only one, you can omit.
    prefix = df["prefix"].iloc[0]
    n = int(df["n"].mode().iloc[0])

    ss = strong_scaling(df, metric=metric, prefix=prefix, n=n)
    plot_strong(ss, metric=metric, out=f"strong_{prefix}_N{n}_{metric}_{directory}.png")

    # Weak scaling: requires multiple N values with dof/P approximately constant.
    if df["n"].nunique() > 1:
        ws = weak_scaling(df, metric=metric, prefix=prefix)
        # If you have multiple dof_per_rank groups, filter one:
        # ws = ws[ws["dof_per_rank"].between(1.0e6, 1.1e6)]
        plot_weak(ws, metric=metric, out=f"weak_{prefix}_{metric}_{directory}.png")

    # Optional: strong scaling for a few common PETSc events if present
    for ev in ["KSPSolve", "SNESFunctionEval", "MatMult", "PCApply"]:
        if (df["metric"] == ev).any():
            ss_ev = strong_scaling(df, metric=ev, prefix=prefix, n=n)
            plot_strong(ss_ev, metric=ev, out=f"strong_{prefix}_N{n}_{ev}.png")

    plt.show()