#!/usr/bin/env bash
# Replace the OW03/CE03 comparison jobs that used the self-trained ce_score_2 filter (team dropped self-training).
for pat in "sm_jobs/66_ow03_compare" "sm_jobs/68_ce03_compare"; do
  for pid in $(pgrep -f "$pat"); do kill -- -"$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill "$pid" 2>/dev/null; done
done
sleep 2; pgrep -fa "_compare" || echo "old compare jobs stopped"
