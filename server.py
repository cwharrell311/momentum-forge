"""
MomentumForge API Server
Flask API to serve screener data to the frontend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from screener import MomentumScreener
import os
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global state
screener = MomentumScreener(fmp_api_key=os.environ.get('FMP_API_KEY'))
scan_results = []
scan_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready to scan',
    'last_scan': None,
    'stocks_found': 0,
    'social_enabled': True
}


def progress_callback(message: str, progress: float):
    """Update scan progress."""
    global scan_status
    scan_status['message'] = message
    scan_status['progress'] = int(progress * 100)


def run_scan_async(min_market_cap: float, include_social: bool = True):
    """Run scan in background thread."""
    global scan_results, scan_status
    
    scan_status['running'] = True
    scan_status['progress'] = 0
    scan_status['message'] = 'Starting scan...'
    scan_status['social_enabled'] = include_social
    
    try:
        results = screener.run_scan(
            min_market_cap_b=min_market_cap,
            include_social=include_social,
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
    include_social = data.get('include_social', True)
    
    # Start scan in background
    thread = threading.Thread(target=run_scan_async, args=(min_market_cap, include_social))
    thread.start()
    
    return jsonify({
        'message': 'Scan started', 
        'min_market_cap': min_market_cap,
        'include_social': include_social
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
            'bullish_sentiment': 0,
            'bullish_social': 0,
            'trending': 0,
            'avg_score': 0
        })
    
    high_conviction = len([r for r in scan_results if r['momentum_score'] >= 80])
    bullish_options = len([r for r in scan_results if r.get('options_volume_ratio', 0) and r['options_volume_ratio'] > 1.5])
    bullish_social = len([r for r in scan_results if r.get('social_score', 0) > 0.2])
    trending = len([r for r in scan_results if r.get('social_trending', False)])
    avg_score = sum(r['momentum_score'] for r in scan_results) / len(scan_results)
    
    return jsonify({
        'total_scanned': 150,  # Universe size
        'signals_found': len(scan_results),
        'high_conviction': high_conviction,
        'bullish_options_flow': bullish_options,
        'bullish_social': bullish_social,
        'trending': trending,
        'avg_score': round(avg_score, 1),
        'last_scan': scan_status.get('last_scan'),
        'social_enabled': scan_status.get('social_enabled', True)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
