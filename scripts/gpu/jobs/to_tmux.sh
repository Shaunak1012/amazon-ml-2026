#!/usr/bin/env bash
# Move the GPU runner into a tmux session (this lab box kills an SSH session's processes on logout; tmux panes live
# in their own systemd scope and survive). Keeps the same queue link and POST policy. nonce: __NONCE__
BASE=$HOME/Desktop/aml
command -v tmux >/dev/null || { echo "NO TMUX on this box"; exit 1; }
tmux -V
OLD=$(pgrep -f scripts/gpu/remote_runner.py | tr '\n' ' ')
Q=$(ps -o args= -p ${OLD%% *} | sed -n 's/.*--queue-url \([^ ]*\).*/\1/p')
[ -n "$Q" ] || { echo "could not read the queue URL from the running runner"; exit 1; }
tmux kill-session -t gpurun 2>/dev/null
tmux new-session -d -s gpurun "cd $BASE/repo && exec .venv/bin/python scripts/gpu/remote_runner.py --queue-url '$Q' --post $BASE/post.json >> $BASE/runner.out 2>&1"
sleep 5
NEW=$(tmux list-panes -t gpurun -F '#{pane_pid}')
echo "tmux runner pid $NEW; old runner pids $OLD"
tmux ls
for p in $OLD; do [ "$p" != "$NEW" ] && kill "$p"; done
sleep 2; pgrep -fa scripts/gpu/remote_runner.py | cut -c1-80
cat /proc/$NEW/cgroup 2>/dev/null | tail -2
echo TMUX RUNNER OK
