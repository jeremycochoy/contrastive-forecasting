#!/bin/bash
# Run this once the first training checkpoint has synced down (typically
# after ~5k steps, ~15 min into the first backbone). Confirms by NAME and
# SIZE that every expected file class made it locally — the CLAUDE.md rule
# "verifying the sync means manually checking the files are there".
#
# Usage:
#   bash scripts/verify_first_checkpoint.sh sync_multiexp/checkpoints/
set -u
DIR=${1:-sync_multiexp/checkpoints}

echo "=== Verifying $DIR ==="
echo
echo "Backbone files (.pth, ~80 MB):"
find "$DIR" -maxdepth 1 -name "tiny_*.pth" ! -name "*_optimizer*" ! -name "*.tmp" ! -name "*.prev" -exec ls -la {} \;
echo
echo "Backbone optimizer files (~155 MB):"
find "$DIR" -maxdepth 1 -name "tiny_*_optimizer.pth" ! -name "*.tmp" ! -name "*.prev" -exec ls -la {} \;
echo
echo "Head files (R1*.pth, ~2.4 MB):"
find "$DIR" -maxdepth 1 -name "R1*.pth" ! -name "*_optimizer*" ! -name "*.tmp" ! -name "*.prev" -exec ls -la {} \;
echo
echo "Head optimizer files (~5 MB):"
find "$DIR" -maxdepth 1 -name "R1*_optimizer.pth" ! -name "*.tmp" ! -name "*.prev" -exec ls -la {} \;
echo
echo "Loss CSVs:"
find "$DIR" -maxdepth 1 -name "*_losses.csv" -exec ls -la {} \;
echo
echo "Results CSVs (in sync_multiexp/results/<run>/):"
find "$(dirname "$DIR")/results" -name "all_results.csv" 2>/dev/null -exec ls -la {} \;
