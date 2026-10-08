#!/bin/bash
# Night 30 Sep 2026: waits until the De Pijp store's task statistics (grid 64 and 128) contain the trajectories
# entry (written after the trajectory build, ~02:10), then runs experiments.RUN
# (= c5_overnight_30sep_queue). Start detached:
#   cd harness && nohup ./run_overnight_30sep.sh > ../runs/logs/overnight_30sep.log 2>&1 &
cd "$(dirname "$0")"
S=../data/amsterdam/de_pijp/crops/v4/c4_100k
Z=../data/amsterdam/zuid_centrum/crops/v4ns/c4_512_20k
until grep -q '"trajectories"' $S/c4_task_stats.json 2>/dev/null && grep -q '"trajectories"' $S/c4_task_stats_g128.json 2>/dev/null; do
  echo "$(date +%H:%M) waiting for De Pijp trajectories statistics"; sleep 60
done
until test -f $Z/c5_traj_train.npz && test -f $Z/c5_traj_val.npz; do
  echo "$(date +%H:%M) waiting for Zuid+Centrum c5_traj_train.npz and c5_traj_val.npz"; sleep 60
done
echo "$(date +%H:%M) sidecars ready, starting the harness"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../.venv/bin/python harness.py
echo "$(date +%H:%M) harness finished"
