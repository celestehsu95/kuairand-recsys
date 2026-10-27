#!/usr/bin/env bash
# Start the serving stack:  ./demo.sh            (Redis, Kafka, bootstrap, API, worker)
#                           ./demo.sh replay     (also stream the test-period log as live traffic)
#                           ./demo.sh loadtest   (latency benchmark -> results/latency.json)
#                           ./demo.sh test       (online/offline feature parity test against Redis)
#                           ./demo.sh down
# Use ARTIFACTS=./artifacts_synth ./demo.sh to try the stack on the smoke-test artifacts.
set -euo pipefail
cd "$(dirname "$0")"
ART="${ARTIFACTS:-./artifacts}"

case "${1:-up}" in
  up)
    [ -f "$ART/ranker_rt.pt" ] || { echo "No artifacts in $ART. Run ./run_all.sh (or the Colab notebook) first."; exit 1; }
    mkdir -p results
    docker compose build
    docker compose up -d --wait redis kafka
    docker compose run --rm bootstrap
    docker compose up -d --wait api worker
    echo
    echo "Demo running: http://localhost:8000"
    echo "Live traffic:  ./demo.sh replay     Load test: ./demo.sh loadtest"
    ;;
  replay)
    docker compose --profile replay up -d replayer
    echo "Replaying test-period events. Turn on auto-refresh in the demo to watch feeds change."
    ;;
  loadtest)
    rm -f results/latency.json
    for c in 1 8 32; do
      docker compose run --rm --no-deps api python -m serving.loadtest --url http://api:8000 --concurrency "$c" --duration 20
    done
    docker compose run --rm --no-deps api python -m serving.loadtest --url http://api:8000 --concurrency 8 --duration 20 --no_realtime
    echo "Done. Run 'python make_report.py --update_readme' to add the latency table to README.md"
    ;;
  test)
    docker compose run --rm --no-deps api python tests/test_parity.py
    ;;
  down)
    docker compose --profile replay --profile tools down
    ;;
  *) echo "usage: $0 [up|replay|loadtest|test|down]"; exit 1 ;;
esac
