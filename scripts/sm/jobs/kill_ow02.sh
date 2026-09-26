#!/usr/bin/env bash
# OW02 (CPU owner model) is superseded by OW03 on the GPU: stop it to free the r7i cores.
for pat in "sm_jobs/62_ow02_train" "sm_jobs/63_ow02_test" "er_owner train --dir runs/OW02" "er_owner score --dir runs/OW02"; do
  for pid in $(pgrep -f "$pat"); do kill -- -"$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill "$pid" 2>/dev/null; done
done
sleep 3; pgrep -fa "OW02" || echo "OW02 stopped"
