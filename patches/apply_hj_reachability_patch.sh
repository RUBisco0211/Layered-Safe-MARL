#!/bin/bash
# Apply the hj-reachability 0.5.0 dynamics.py fix.
#
# The installed hj-reachability has a bug: Dynamics.optimal_control() and
# Dynamics.optimal_disturbance() call self.optimal_control_and_disturbance()
# with an extra `self` argument, which breaks safety-filter training
# (TypeError: ... takes 4 positional arguments but 5 were given).
#
# This script reinstalls hj-reachability and patches dynamics.py to remove
# the stray `self`.
#
# Usage:  bash patches/apply_hj_reachability_patch.sh
#   (run from repo root; uses the active `python`/`pip`, e.g. layer-safe-marl)

set -e

echo "=== Reinstalling hj-reachability==0.5.0 ==="
python -m pip install --force-reinstall --no-deps "hj-reachability==0.5.0"

DYN=$(python -c "import hj_reachability, os; print(os.path.join(os.path.dirname(hj_reachability.__file__), 'dynamics.py'))")
echo "=== Patching: $DYN ==="

sed -i 's/optimal_control_and_disturbance(self, state, time, grad_value)/optimal_control_and_disturbance(state, time, grad_value)/g' "$DYN"

echo "=== Verify ==="
grep -n "optimal_control" "$DYN"
