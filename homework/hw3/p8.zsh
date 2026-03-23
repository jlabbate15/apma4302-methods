#!/usr/bin/env zsh
set -euo pipefail

BIN="./reaction2d"

# For DMDA uniform refinement with ratio 2:
# new_grid = (base_grid - 1) * 2^refine + 1
# Choose base_grid=9 so refine=2..6 gives 33, 65, 129, 257, 513.
BASE_GRID=9

procs=(1 2 4)
refines=(2 3 4 5 6)

# Nonlinear problem: gamma=100, p=3, and DO NOT set -rct_linear_f
common_flags=(
  -rct_gamma 100
  -rct_p 3
  -rct_linear_f false
  -ksp_rtol 1.e-12
  -ksp_atol 1.e-14
  -ksp_monitor
  -snes_monitor
  -log_view
)

# If you instead need the linearized RHS (-rct_linear_f true), uncomment:
# common_flags+=( -rct_linear_f true )

for nprocs in $procs; do
  for r in $refines; do
    outdir="runs/p8_n${nprocs}_ref${r}"
    mkdir -p "$outdir"

    log="$outdir/run.log"
    echo "Running nprocs=$nprocs refine=$r -> $log"

    mpirun -n "$nprocs" "$BIN" \
      "${common_flags[@]}" \
      -da_grid_x "$BASE_GRID" -da_grid_y "$BASE_GRID" \
      -da_refine "$r" \
      >"$log" 2>&1

    # Keep the VTK output from each run (the code always writes reaction2d.vtr)
    if [[ -f "reaction2d.vtr" ]]; then
      cp -f "reaction2d.vtr" "$outdir/reaction2d_ref${r}.vtr"
    fi
  done
done