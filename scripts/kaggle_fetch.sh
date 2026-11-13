#!/bin/sh
# Fetch finished Kaggle kernels into runs/ (skips kernels already fetched or still running).
#   sh scripts/kaggle_fetch.sh            # from the repository root
SLUGS="rlpd-t4-lh-demos-rlpd-s01 rlpd-t4-lh-demos-rlpd-s23 rlpd-t4-lh-demos-rlpd-s4 rlpd-t4-lh-1tour-s01 rlpd-t4-lh-1tour-s23 rlpd-t4-lh-1tour-s4 rlpd-t3-lh-s02 rlpd-t3-lh-s34 rlpd-t4-4pistes-s02 rlpd-t4-4pistes-s34 rlpd-t4-lh-stats16-s34 rlpd-t4-lh-stats16-s02 rlpd-t4-snow-s34 rlpd-t4-snow-s02 rlpd-t4-16pistes-s01 rlpd-t4-16pistes-s23 rlpd-t4-16pistes-s4 sac-t4-lh-s01 sac-t4-lh-s23 sac-t4-lh-s4 rlpd-t4-16pistes-s2b"
mkdir -p ../kaggle_out
for k in $SLUGS; do
  st=$(kaggle kernels status "matleniz/$k" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | cut -d. -f2)
  if [ -d "../kaggle_out/$k" ]; then echo "$k: fetched"; continue; fi
  echo "$k: ${st:-not pushed}"
  [ "$st" = COMPLETE ] || [ "$st" = ERROR ] || continue
  kaggle kernels output "matleniz/$k" -p "../kaggle_out/$k" >/dev/null 2>&1
  [ -d "../kaggle_out/$k/runs/rlpd" ] && cp -r --update=none "../kaggle_out/$k/runs/rlpd/." runs/rlpd/
done
