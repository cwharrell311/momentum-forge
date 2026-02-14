#!/bin/bash
# ============================================================
#   SPY DAYTRADING SIGNAL SCANNER
#
#   Usage:
#     ./go.sh              # Today's signals
#     ./go.sh backtest     # Full 5-year backtest
#     ./go.sh optimize     # Backtest + parameter optimization
# ============================================================

set -e
cd "$(dirname "$0")"

# Install deps quietly if missing
pip3 install -q yfinance matplotlib tabulate 2>/dev/null || true

MODE="${1:-signals}"

case "$MODE" in
    signals|today|scan|"")
        PYTHONPATH=. python3 -m backtest.signals
        ;;
    backtest|test|bt)
        PYTHONPATH=. python3 -m backtest.run --years "${2:-5}"
        ;;
    optimize|opt)
        PYTHONPATH=. python3 -m backtest.run --years "${2:-5}" --optimize
        ;;
    *)
        echo ""
        echo "  Usage:"
        echo "    ./go.sh              Today's trading signals"
        echo "    ./go.sh backtest     Full 5-year backtest"
        echo "    ./go.sh backtest 3   3-year backtest"
        echo "    ./go.sh optimize     Backtest + parameter optimization"
        echo ""
        ;;
esac
