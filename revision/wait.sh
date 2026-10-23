#!/usr/bin/env bash
# Wait until every named kernel has left QUEUED/RUNNING. Usage: revision/wait.sh a b [minutes_between_checks]
owner=${KAGGLE_OWNER:-devgurucodes}
sessions=("$@")
while :; do
  line=""; busy=0
  for s in "${sessions[@]}"; do
    st=$(python -m kaggle kernels status "$owner/rsl-session-$s" 2>&1 | grep -oE 'QUEUED|RUNNING|COMPLETE|ERROR|CANCEL[A-Z_]*' | head -1)
    line+="$s=$st "
    [[ "$st" == QUEUED || "$st" == RUNNING || -z "$st" ]] && busy=1
  done
  echo "$(date +%H:%M) $line"
  [ $busy = 0 ] && break
  sleep 120
done
