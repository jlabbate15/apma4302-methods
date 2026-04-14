import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

#!/usr/bin/env python3


FLOAT_RE = r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
TIME_PATTERNS = [
    re.compile(rf"\btime\b\s*[=:]?\s*{FLOAT_RE}"),
    re.compile(rf"\bt\b\s*[=:]\s*{FLOAT_RE}"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run petsc executable with mpiexec, pass extra PETSc args, and build a .pvd file "
            "from produced VTK snapshots."
        )
    )
    p.add_argument("-e", "--petsc_exec", help="PETSc executable to run")
    p.add_argument("-n", "--np", type=int, default=1, help="Number of MPI ranks")
    p.add_argument(
        "-o",
        "--options_file",
        default=None,
        help="PETSc options file for -options_file",
    )
    p.add_argument(
        "-f",
        "--filename",
        default="sol",
        help="VTK filename prefix used as <prefix>-%%03D.vts",
    )
    
    p.add_argument(
        "--pvd",
        default=None,
        help="Output .pvd path (default: <filename>.pvd)",
    )
    p.add_argument(
        "petsc_args",
        nargs=argparse.REMAINDER,
        help="Additional PETSc args (optionally after --)",
    )
    return p.parse_args()


def extract_times(lines):
    times = []
    for line in lines:
        for pat in TIME_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    times.append(float(m.group(1)))
                except ValueError:
                    pass
                break
    # De-duplicate consecutive repeated times
    deduped = []
    for t in times:
        if not deduped or abs(t - deduped[-1]) > 1e-14:
            deduped.append(t)
    return deduped


def build_pvd(vts_files, times, pvd_path):
    pvd_path = Path(pvd_path)
    base_dir = pvd_path.parent.resolve()

    # Align times to number of files
    n = len(vts_files)
    if len(times) < n:
        times = times + [float(i) for i in range(len(times), n)]
    elif len(times) > n:
        times = times[:n]

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for i, vf in enumerate(vts_files):
        rel = os.path.relpath(Path(vf).resolve(), base_dir)
        lines.append(
            f'    <DataSet timestep="{times[i]:.16g}" group="" part="0" file="{rel}"/>'
        )
    lines += ["  </Collection>", "</VTKFile>"]

    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    extra = list(args.petsc_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    print("Running:", " ".join([str(args.petsc_exec)] + extra), flush=True)

    vtk_pattern = f"{args.filename}-%03D.vts"

    cmd = [
        "mpiexec",
        "-np",
        str(args.np),
        str(args.petsc_exec)]
    if args.options_file:
        cmd += ["-options_file", str(args.options_file)]
    cmd += ["-ts_monitor_solution_vtk", vtk_pattern, "-ts_monitor"] + extra
    print(cmd)


    print("Running:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        output_lines.append(line.rstrip("\n"))

    '''ret = proc.wait()
    if ret != 0:
        sys.exit(ret)
    '''

    # Find generated VTK files
    vts_files = sorted(glob.glob(f"{args.filename}-*.vts"))
    if not vts_files:
        print("No VTS files found; skipping .pvd generation.", file=sys.stderr)
        return

    times = extract_times(output_lines)
    pvd_path = args.pvd if args.pvd else f"{args.filename}.pvd"
    build_pvd(vts_files, times, pvd_path)
    print(f"Wrote {pvd_path} with {len(vts_files)} entries.")


if __name__ == "__main__":
    main()