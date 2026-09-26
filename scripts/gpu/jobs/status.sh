#!/usr/bin/env bash
# GPU box status: processes, GPU, and the tail of every job log (including jobs orphaned by a runner restart).
# nonce: __NONCE__
date
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "== python processes"; pgrep -fa "python" | grep -v remote_runner | cut -c1-200
for f in runs/gpu_logs/*.log; do echo "== $f ($(wc -c < "$f") bytes)"; tail -12 "$f"; done
ls -la runs/OW03 2>/dev/null | head; ls data/cache 2>/dev/null
