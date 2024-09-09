#!/bin/bash
# Bandit programs must exit with code 0 and emit the number of failures.
# This accepts a percentage of calls to fail and emits a 0/1 according to that percentage.

if [[ $# -ne 2 ]]
then
  echo "USAGE '$0 <interestingness likelyhood (0.0, 1.0)> <sleep time>'"
  exit 124
fi

sleep $2

if [[ $(awk -v n=1 -v seed="$RANDOM" 'BEGIN { srand(seed); printf("%.4f\n", rand()) }'| tr -d .| sed 's/^0*//') -le $(printf "%.4f\n" $1 | tr -d .| sed 's/^0*//') ]]; then
  echo "Interesting case"
  exit 1
else
  echo "Uninteresting case"
  exit 0
fi
