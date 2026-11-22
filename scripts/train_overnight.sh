#!/usr/bin/env bash
# Træn den store model natten over (ca. 6,5 timer på en M3 Pro).
#
#   ./scripts/train_overnight.sh
#
# Hvad scriptet gør:
#   1. flytter de nuværende data/ og runs/ til archive/, så intet går tabt
#   2. laver et nyt datasæt med de længere PRD'er
#   3. træner koncept-tokenizer (45 min), generator (250 min) og baseline
#      (lige så mange skridt som generatoren, ca. 95 min)
#   4. evaluerer NCM mod baseline og gemmer runs/results.json
#
# Alt bliver logget til runs/overnight.log. `caffeinate` holder Macen vågen.
# Husk: lad opladeren sidde i, og luk ikke låget.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

stamp=$(date +%Y%m%d-%H%M)
mkdir -p archive
[ -d runs ] && mv runs "archive/runs-$stamp"
[ -d data ] && mv data "archive/data-$stamp"
mkdir -p runs

caffeinate -i bash -c '
  set -euo pipefail
  echo "Start: $(date)"
  tinyprd data
  tinyprd train-all
  tinyprd evaluate
  echo "Færdig: $(date)"
' 2>&1 | tee runs/overnight.log
