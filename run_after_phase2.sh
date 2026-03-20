#!/bin/bash
# Wait for Phase 2 to finish, then launch Phase 4
set -e
cd ~/workspaces/contrastive-forecasting

echo "Waiting for Phase 2 to complete..."
while pgrep -f "train_contrastive_v2.*T[67]" > /dev/null 2>&1; do
    sleep 60
done
echo "Phase 2 done at $(date)"

# Print Phase 2 summary
echo ""
echo "=== Phase 2 Final Results ==="
python3 -c "
import json, glob
for f in sorted(glob.glob('arch_search_T*_results.json')):
    d = json.load(open(f))
    fm = d.get('final_metrics', {})
    eid = d['experiment_id']
    print(eid, 'FF=%.4f' % fm.get('val_ff',0), 'gap=%.4f' % fm.get('val_ff_fp_gap',0), 'CB=%.4f' % fm.get('val_cb',0))
"

echo ""
echo "Launching Phase 4..."
bash run_phase4.sh
