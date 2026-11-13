#!/bin/sh
# Fetch finished Kaggle kernels every 10 min, rebuild the registry and commit runs/ (local commits only).
# Stops when the 4 kernels relaunched on 13/11 are fetched, or if one of them ends in ERROR.
#   setsid nohup sh scripts/kaggle_collect.sh > runs/logs/kaggle_collect.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
NEW="sac-t4-lh-s01 sac-t4-lh-s23 sac-t4-lh-s4 rlpd-t4-16pistes-s2b"
while :; do
  out=$(sh scripts/kaggle_fetch.sh 2>&1)
  ../.venv/bin/python -m lab.registry >/dev/null 2>&1
  if [ -n "$(git status --short runs)" ]; then
    git add runs && git commit -qm "Add fetched RLPD runs (Kaggle and local)

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Tuvr72W68ZRs59pGiqAMvc"
  fi
  n=0; for k in $NEW; do [ -d "../kaggle_out/$k" ] && n=$((n + 1)); echo "$out" | grep -q "^$k: ERROR" && { echo "KERNEL ERROR $k"; exit 1; }; done
  [ "$n" -ge 4 ] && break
  sleep 600
done
date; echo "$out"
