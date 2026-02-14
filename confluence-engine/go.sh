#!/bin/bash
# ============================================================
#   SPY DAYTRADING SIGNAL SCANNER
#
#   Usage:
#     ./go.sh              # Today's signals
#     ./go.sh backtest     # Full 5-year backtest
#     ./go.sh optimize     # Backtest + parameter optimization
#     ./go.sh data download    # Download real SPY+VIX data (run on your Mac)
#     ./go.sh data import SPY.csv  # Import CSV from Yahoo Finance
#     ./go.sh data status      # Show cached data info
#     ./go.sh data clear       # Wipe cache
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
    results|summary|last)
        PYTHONPATH=. python3 -m backtest.run --results
        ;;
    data)
        SUBCMD="${2:-status}"
        case "$SUBCMD" in
            download|dl)
                PYTHONPATH=. python3 -m backtest.run --download --years "${3:-5}"
                ;;
            import)
                # Import CSV files: ./go.sh data import SPY.csv [VIX.csv]
                SPY_CSV="${3}"
                VIX_CSV="${4}"
                if [ -z "$SPY_CSV" ]; then
                    echo ""
                    echo "  Usage: ./go.sh data import <SPY.csv> [VIX.csv]"
                    echo ""
                    echo "  Download from Yahoo Finance:"
                    echo "    1. Go to finance.yahoo.com/quote/SPY/history"
                    echo "    2. Set time period to 5Y, click Download"
                    echo "    3. Run: ./go.sh data import ~/Downloads/SPY.csv"
                    echo ""
                    echo "  For VIX (optional but recommended):"
                    echo "    1. Go to finance.yahoo.com/quote/%5EVIX/history"
                    echo "    2. Download, then: ./go.sh data import SPY.csv VIX.csv"
                    echo ""
                    exit 1
                fi
                ARGS="--import-spy $SPY_CSV"
                if [ -n "$VIX_CSV" ]; then
                    ARGS="$ARGS --import-vix $VIX_CSV"
                fi
                PYTHONPATH=. python3 -m backtest.run $ARGS --years "${5:-5}"
                ;;
            status|info)
                PYTHONPATH=. python3 -m backtest.run --data-status
                ;;
            clear|clean|reset)
                PYTHONPATH=. python3 -m backtest.run --clear-cache
                ;;
            *)
                echo ""
                echo "  Data commands:"
                echo "    ./go.sh data download      Download real SPY+VIX data via yfinance"
                echo "    ./go.sh data import F.csv   Import CSV from Yahoo Finance"
                echo "    ./go.sh data status         Show what's cached"
                echo "    ./go.sh data clear          Wipe cache"
                echo ""
                ;;
        esac
        ;;
    *)
        echo ""
        echo "  Usage:"
        echo "    ./go.sh              Today's trading signals"
        echo "    ./go.sh backtest     Full 5-year backtest"
        echo "    ./go.sh backtest 3   3-year backtest"
        echo "    ./go.sh optimize     Backtest + parameter optimization"
        echo "    ./go.sh results      Show last backtest results summary"
        echo ""
        echo "  Data management:"
        echo "    ./go.sh data download      Download real SPY+VIX data"
        echo "    ./go.sh data import F.csv  Import from Yahoo Finance CSV"
        echo "    ./go.sh data status        Show cached data"
        echo "    ./go.sh data clear         Clear cached data"
        echo ""
        ;;
esac
