"""
MomentumForge API Server
Flask API to serve screener data to the frontend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from screener import MomentumScreener
from congress_tracker import CongressTracker
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global state
fmp_key = os.environ.get('FMP_API_KEY')
screener = MomentumScreener(fmp_api_key=fmp_key)
congress_tracker = CongressTracker(fmp_api_key=fmp_key)
congress_trades = []
congress_loading = False
scan_results = []
scan_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready to scan',
    'last_scan': None,
    'stocks_found': 0
}


def progress_callback(message: str, progress: float):
    """Update scan progress."""
    global scan_status
    scan_status['message'] = message
    scan_status['progress'] = int(progress * 100)


def run_scan_async(min_market_cap: float):
    """Run scan in background thread."""
    global scan_results, scan_status

    scan_status['running'] = True
    scan_status['progress'] = 0
    scan_status['message'] = 'Starting scan...'

    try:
        results = screener.run_scan(
            min_market_cap_b=min_market_cap,
            progress_callback=progress_callback
        )

        scan_results = [r.to_dict() for r in results]
        scan_status['stocks_found'] = len(results)
        scan_status['last_scan'] = datetime.now().isoformat()
        scan_status['message'] = f'Scan complete. Found {len(results)} stocks.'
        scan_status['progress'] = 100

    except Exception as e:
        scan_status['message'] = f'Error: {str(e)}'

    finally:
        scan_status['running'] = False


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/status', methods=['GET'])
def status():
    """Get current scan status."""
    return jsonify(scan_status)


@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start a new scan."""
    global scan_status

    if scan_status['running']:
        return jsonify({'error': 'Scan already in progress'}), 400

    data = request.get_json() or {}
    min_market_cap = data.get('min_market_cap', 1.0)

    # Start scan in background
    thread = threading.Thread(target=run_scan_async, args=(min_market_cap,))
    thread.start()

    return jsonify({
        'message': 'Scan started',
        'min_market_cap': min_market_cap
    })


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get scan results with optional filtering."""
    
    # Get filter parameters
    min_score = request.args.get('min_score', 0, type=int)
    sector = request.args.get('sector', 'all')
    signal = request.args.get('signal', 'all')
    limit = request.args.get('limit', 100, type=int)
    
    # Filter results
    filtered = scan_results.copy()
    
    if min_score > 0:
        filtered = [r for r in filtered if r['momentum_score'] >= min_score]
    
    if sector != 'all':
        filtered = [r for r in filtered if r['sector'] == sector]
    
    if signal != 'all':
        filtered = [r for r in filtered if signal in r['signals']]
    
    # Limit results
    filtered = filtered[:limit]
    
    return jsonify({
        'count': len(filtered),
        'total': len(scan_results),
        'results': filtered
    })


@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock(ticker: str):
    """Get detailed data for a specific stock."""
    
    # Check if in results
    for stock in scan_results:
        if stock['ticker'].upper() == ticker.upper():
            return jsonify(stock)
    
    # If not in results, analyze it directly
    result = screener.analyze_stock(ticker.upper())
    if result:
        return jsonify(result.to_dict())
    
    return jsonify({'error': 'Stock not found or does not meet criteria'}), 404


@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    """Get list of unique sectors in results."""
    sectors = list(set(r['sector'] for r in scan_results if r.get('sector')))
    sectors.sort()
    return jsonify(sectors)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get summary statistics."""
    if not scan_results:
        return jsonify({
            'total_scanned': 0,
            'signals_found': 0,
            'high_conviction': 0,
            'bullish_options_flow': 0,
            'avg_score': 0
        })

    high_conviction = len([r for r in scan_results if r['momentum_score'] >= 60])
    bullish_options = len([r for r in scan_results if r.get('options_volume_ratio', 0) and r['options_volume_ratio'] > 1.5])
    avg_score = sum(r['momentum_score'] for r in scan_results) / len(scan_results)

    return jsonify({
        'total_scanned': len(scan_results),
        'signals_found': len(scan_results),
        'high_conviction': high_conviction,
        'bullish_options_flow': bullish_options,
        'avg_score': round(avg_score, 1),
        'last_scan': scan_status.get('last_scan')
    })


# ==========================================
# Congressional Trade Tracker Endpoints
# ==========================================

@app.route('/api/congress/trades', methods=['GET'])
def get_congress_trades():
    """Get congressional stock trades."""
    global congress_trades, congress_loading

    # Filters
    chamber = request.args.get('chamber', 'all')
    party = request.args.get('party', 'all')
    trade_type = request.args.get('trade_type', 'all')
    ticker = request.args.get('ticker', '')
    politician = request.args.get('politician', '')

    # Fetch if empty
    data_source = None
    source_errors = []
    if not congress_trades and not congress_loading:
        congress_loading = True
        try:
            trades = congress_tracker.get_trades(days_back=365)
            congress_trades = [t.to_dict() for t in trades]
            data_source = congress_tracker.source_used
            source_errors = congress_tracker.source_errors
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            congress_loading = False
    else:
        data_source = congress_tracker.source_used
        source_errors = congress_tracker.source_errors

    filtered = congress_trades.copy()

    if chamber != 'all':
        filtered = [t for t in filtered if t['chamber'] == chamber]

    if party != 'all':
        filtered = [t for t in filtered if t['party'] == party]

    if trade_type != 'all':
        if trade_type == 'Purchase':
            filtered = [t for t in filtered if 'Purchase' in t['trade_type']]
        elif trade_type == 'Sale':
            filtered = [t for t in filtered if 'Sale' in t['trade_type']]

    if ticker:
        filtered = [t for t in filtered if t['ticker'].upper() == ticker.upper()]

    if politician:
        pol_lower = politician.lower()
        filtered = [t for t in filtered if pol_lower in t['politician'].lower()]

    return jsonify({
        'count': len(filtered),
        'total': len(congress_trades),
        'trades': filtered,
        'data_source': data_source,
        'source_errors': source_errors,
    })


@app.route('/api/congress/refresh', methods=['POST'])
def refresh_congress_trades():
    """Force refresh congressional trades data."""
    global congress_trades, congress_loading

    if congress_loading:
        return jsonify({'error': 'Already loading'}), 400

    congress_loading = True

    def fetch_async():
        global congress_trades, congress_loading
        try:
            congress_tracker._cache = None  # Clear cache
            trades = congress_tracker.get_trades(days_back=365)
            congress_trades = [t.to_dict() for t in trades]
        except Exception as e:
            print(f"Congress refresh error: {e}")
        finally:
            congress_loading = False

    thread = threading.Thread(target=fetch_async)
    thread.start()

    return jsonify({'message': 'Refresh started'})


@app.route('/api/congress/stats', methods=['GET'])
def get_congress_stats():
    """Get congressional trading statistics."""
    if not congress_trades:
        return jsonify({
            'total_trades': 0,
            'total_politicians': 0,
            'total_purchases': 0,
            'total_sales': 0,
            'top_traded_tickers': [],
            'most_active_politicians': [],
            'avg_days_to_disclose': 0,
        })

    purchases = [t for t in congress_trades if 'Purchase' in t.get('trade_type', '')]
    sales = [t for t in congress_trades if 'Sale' in t.get('trade_type', '')]

    ticker_counts = {}
    for t in congress_trades:
        tk = t.get('ticker', '')
        ticker_counts[tk] = ticker_counts.get(tk, 0) + 1
    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    pol_counts = {}
    for t in congress_trades:
        p = t.get('politician', '')
        pol_counts[p] = pol_counts.get(p, 0) + 1
    top_pols = sorted(pol_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    disclosure_days = [t['days_to_disclose'] for t in congress_trades if t.get('days_to_disclose', 0) > 0]
    avg_days = sum(disclosure_days) / len(disclosure_days) if disclosure_days else 0

    return jsonify({
        'total_trades': len(congress_trades),
        'total_politicians': len(set(t.get('politician', '') for t in congress_trades)),
        'total_purchases': len(purchases),
        'total_sales': len(sales),
        'top_traded_tickers': [{'ticker': t, 'count': c} for t, c in top_tickers],
        'most_active_politicians': [{'name': n, 'count': c} for n, c in top_pols],
        'avg_days_to_disclose': round(avg_days, 1),
    })


@app.route('/api/congress/status', methods=['GET'])
def congress_status():
    """Check if congress data is loading."""
    return jsonify({
        'loading': congress_loading,
        'has_data': len(congress_trades) > 0,
        'trade_count': len(congress_trades),
        'data_source': congress_tracker.source_used,
        'source_errors': congress_tracker.source_errors,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
