"""
Operator Toolbox Helper Module

This module provides helper functions for the Operator Toolbox feature,
including data health checks, cache rebuilding, and self-testing capabilities.

All functions are designed to be safe with comprehensive try/except blocks
and clear error reporting for the UI.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import traceback

# Add parent helpers directory to path
helpers_dir = os.path.dirname(os.path.abspath(__file__))
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

logger = logging.getLogger(__name__)


def get_data_health_metadata() -> Dict[str, Any]:
    """
    Gather comprehensive data health metadata for display in the Operator Toolbox.
    
    Returns:
        Dictionary containing:
        - last_trading_day: From price_book index (if available)
        - price_book_max_date: Max date in price cache
        - wave_history_max_date: Max date from wave_history (if exists)
        - required_symbols_present: Dict with presence checks for SPY/QQQ/IWM, VIX variants, T-bill variants
        - missing_tickers: List of missing tickers
        - stale_tickers: List of stale tickers (if detectable)
        - vix_overlay_status: Dict with VIX overlay status information
        - errors: List of any errors encountered
    """
    metadata = {
        'last_trading_day': None,
        'price_book_max_date': None,
        'wave_history_max_date': None,
        'required_symbols_present': {
            'benchmarks': {'SPY': False, 'QQQ': False, 'IWM': False},
            'vix_any': False,
            'tbill_any': False
        },
        'missing_tickers': [],
        'stale_tickers': [],
        'vix_overlay_status': {
            'is_live': False,
            'config_available': False,
            'message': 'Unknown'
        },
        'errors': []
    }
    
    # Get price_book data
    try:
        import price_book
        price_data = price_book.get_price_book()
        
        if not price_data.empty:
            # Get last trading day from index
            metadata['last_trading_day'] = price_data.index[-1].strftime('%Y-%m-%d')
            metadata['price_book_max_date'] = price_data.index[-1].strftime('%Y-%m-%d')
            
            # Check required symbols presence
            available_tickers = set(price_data.columns)
            
            # Check benchmarks (ALL required)
            for ticker in ['SPY', 'QQQ', 'IWM']:
                metadata['required_symbols_present']['benchmarks'][ticker] = ticker in available_tickers
            
            # Check VIX variants (ANY required)
            vix_variants = ['^VIX', 'VIXY', 'VXX']
            metadata['required_symbols_present']['vix_any'] = any(v in available_tickers for v in vix_variants)
            
            # Check T-bill variants (ANY required)
            tbill_variants = ['BIL', 'SHY']
            metadata['required_symbols_present']['tbill_any'] = any(t in available_tickers for t in tbill_variants)
            
    except Exception as e:
        metadata['errors'].append(f"Error reading price_book: {str(e)}")
        logger.error(f"Error reading price_book: {e}", exc_info=True)
    
    # Get wave_history max date
    try:
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        if os.path.exists(wave_history_path):
            wave_history = pd.read_csv(wave_history_path)
            if 'date' in wave_history.columns and not wave_history.empty:
                wave_history['date'] = pd.to_datetime(wave_history['date'])
                metadata['wave_history_max_date'] = wave_history['date'].max().strftime('%Y-%m-%d')
    except Exception as e:
        metadata['errors'].append(f"Error reading wave_history: {str(e)}")
        logger.error(f"Error reading wave_history: {e}", exc_info=True)
    
    # Get missing/stale tickers
    try:
        # Check for missing_tickers.csv if it exists
        missing_tickers_path = os.path.join(os.getcwd(), 'missing_tickers.csv')
        if os.path.exists(missing_tickers_path):
            missing_df = pd.read_csv(missing_tickers_path)
            if 'ticker' in missing_df.columns:
                metadata['missing_tickers'] = missing_df['ticker'].tolist()
    except Exception as e:
        metadata['errors'].append(f"Error reading missing_tickers: {str(e)}")
        logger.error(f"Error reading missing_tickers: {e}", exc_info=True)
    
    # Get VIX overlay status
    try:
        from config.vix_overlay_config import get_vix_overlay_status, is_vix_overlay_live
        
        vix_status = get_vix_overlay_status()
        metadata['vix_overlay_status'] = {
            'is_live': vix_status['is_live'],
            'config_available': True,
            'resilient_mode': vix_status['resilient_mode'],
            'fallback_vix_level': vix_status['fallback_vix_level'],
            'message': '🟢 LIVE and Active' if vix_status['is_live'] else '⚪ Configured but Disabled'
        }
    except ImportError:
        metadata['vix_overlay_status'] = {
            'is_live': False,
            'config_available': False,
            'message': '⚠️ Config module not available'
        }
    except Exception as e:
        metadata['vix_overlay_status'] = {
            'is_live': False,
            'config_available': False,
            'message': f'❌ Error: {str(e)}'
        }
        logger.error(f"Error checking VIX overlay status: {e}", exc_info=True)
    
    return metadata


def rebuild_price_cache() -> Tuple[bool, str]:
    """
    Rebuild the price cache (prices_cache.parquet) using existing build script.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        cache_path = os.path.join(os.getcwd(), 'data', 'cache', 'prices_cache.parquet')
        
        # Use build_price_cache.py script
        build_script = os.path.join(os.getcwd(), 'build_price_cache.py')
        
        if not os.path.exists(build_script):
            return False, f"Build script not found: {build_script}"
        
        # Run the build script
        logger.info("Starting price cache rebuild...")
        result = subprocess.run(
            [sys.executable, build_script, '--force'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Check if cache was created/updated
            if os.path.exists(cache_path):
                file_stat = os.stat(cache_path)
                size_kb = file_stat.st_size / 1024
                mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                return True, f"✅ Price cache rebuilt successfully!\n\nFile: {cache_path}\nSize: {size_kb:.1f} KB\nUpdated: {mod_time}"
            else:
                return False, f"Build script completed but cache file not found at {cache_path}"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"Build script failed:\n{error_msg[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, "Build timed out after 5 minutes"
    except Exception as e:
        logger.error(f"Error rebuilding price cache: {e}", exc_info=True)
        return False, f"Error: {str(e)}\n\n{traceback.format_exc()[:500]}"


def rebuild_wave_history() -> Tuple[bool, str]:
    """
    Rebuild wave_history.csv from price_book and wave registry.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        
        # Use build_wave_history_from_prices.py script
        build_script = os.path.join(os.getcwd(), 'build_wave_history_from_prices.py')
        
        if not os.path.exists(build_script):
            return False, f"Build script not found: {build_script}"
        
        # Run the build script
        logger.info("Starting wave_history rebuild...")
        result = subprocess.run(
            [sys.executable, build_script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Check if wave_history was created/updated
            if os.path.exists(wave_history_path):
                file_stat = os.stat(wave_history_path)
                size_kb = file_stat.st_size / 1024
                mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Count rows
                try:
                    wave_history = pd.read_csv(wave_history_path)
                    row_count = len(wave_history)
                    wave_count = wave_history['wave'].nunique() if 'wave' in wave_history.columns else 0
                    
                    return True, f"✅ wave_history.csv rebuilt successfully!\n\nFile: {wave_history_path}\nSize: {size_kb:.1f} KB\nRows: {row_count:,}\nWaves: {wave_count}\nUpdated: {mod_time}"
                except Exception:
                    return True, f"✅ wave_history.csv rebuilt successfully!\n\nFile: {wave_history_path}\nSize: {size_kb:.1f} KB\nUpdated: {mod_time}"
            else:
                return False, f"Build script completed but wave_history.csv not found"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"Build script failed:\n{error_msg[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, "Build timed out after 5 minutes"
    except Exception as e:
        logger.error(f"Error rebuilding wave_history: {e}", exc_info=True)
        return False, f"Error: {str(e)}\n\n{traceback.format_exc()[:500]}"


def run_self_test() -> Dict[str, Any]:
    """
    Run internal self-test suite to validate system health.
    
    Returns:
        Dictionary with test results:
        - overall_status: 'PASS' or 'FAIL'
        - tests: List of individual test results
        - summary: Human-readable summary
    """
    test_results = {
        'overall_status': 'PASS',
        'tests': [],
        'summary': '',
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    # Test 1: Check if canonical_data module is importable
    try:
        import canonical_data
        test_results['tests'].append({
            'name': 'Import canonical_data',
            'status': 'PASS',
            'message': 'Module imported successfully'
        })
    except Exception as e:
        test_results['tests'].append({
            'name': 'Import canonical_data',
            'status': 'FAIL',
            'message': f'Failed to import: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 2: Check if wave_registry module is importable
    try:
        import wave_registry
        test_results['tests'].append({
            'name': 'Import wave_registry',
            'status': 'PASS',
            'message': 'Module imported successfully'
        })
    except Exception as e:
        test_results['tests'].append({
            'name': 'Import wave_registry',
            'status': 'FAIL',
            'message': f'Failed to import: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 3: Check if return_pipeline module is importable
    try:
        import return_pipeline
        test_results['tests'].append({
            'name': 'Import return_pipeline',
            'status': 'PASS',
            'message': 'Module imported successfully'
        })
    except Exception as e:
        test_results['tests'].append({
            'name': 'Import return_pipeline',
            'status': 'FAIL',
            'message': f'Failed to import: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 4: Check price cache exists
    try:
        cache_path = os.path.join(os.getcwd(), 'data', 'cache', 'prices_cache.parquet')
        if os.path.exists(cache_path):
            test_results['tests'].append({
                'name': 'Price cache exists',
                'status': 'PASS',
                'message': f'Found at {cache_path}'
            })
        else:
            test_results['tests'].append({
                'name': 'Price cache exists',
                'status': 'FAIL',
                'message': f'Not found at {cache_path}'
            })
            test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'Price cache exists',
            'status': 'FAIL',
            'message': f'Error: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 5: Check wave_history.csv exists
    try:
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        if os.path.exists(wave_history_path):
            test_results['tests'].append({
                'name': 'wave_history.csv exists',
                'status': 'PASS',
                'message': f'Found at {wave_history_path}'
            })
        else:
            test_results['tests'].append({
                'name': 'wave_history.csv exists',
                'status': 'FAIL',
                'message': f'Not found at {wave_history_path}'
            })
            test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'wave_history.csv exists',
            'status': 'FAIL',
            'message': f'Error: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 6: Check wave registry file exists
    try:
        registry_path = os.path.join(os.getcwd(), 'data', 'wave_registry.csv')
        if os.path.exists(registry_path):
            test_results['tests'].append({
                'name': 'Wave registry exists',
                'status': 'PASS',
                'message': f'Found at {registry_path}'
            })
        else:
            test_results['tests'].append({
                'name': 'Wave registry exists',
                'status': 'FAIL',
                'message': f'Not found at {registry_path}'
            })
            test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'Wave registry exists',
            'status': 'FAIL',
            'message': f'Error: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 7: Try to load price data
    try:
        import price_book
        price_data = price_book.get_price_book()
        if not price_data.empty:
            rows, cols = price_data.shape
            test_results['tests'].append({
                'name': 'Load price data',
                'status': 'PASS',
                'message': f'Loaded {rows} days × {cols} tickers'
            })
        else:
            test_results['tests'].append({
                'name': 'Load price data',
                'status': 'FAIL',
                'message': 'Price data is empty'
            })
            test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'Load price data',
            'status': 'FAIL',
            'message': f'Error: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Test 8: Try to load wave registry
    try:
        import wave_registry
        registry = wave_registry.get_wave_registry()
        if not registry.empty:
            wave_count = len(registry)
            active_count = len(registry[registry['active']]) if 'active' in registry.columns else 0
            test_results['tests'].append({
                'name': 'Load wave registry',
                'status': 'PASS',
                'message': f'Loaded {wave_count} waves ({active_count} active)'
            })
        else:
            test_results['tests'].append({
                'name': 'Load wave registry',
                'status': 'FAIL',
                'message': 'Wave registry is empty'
            })
            test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'Load wave registry',
            'status': 'FAIL',
            'message': f'Error: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Generate summary
    passed = sum(1 for t in test_results['tests'] if t['status'] == 'PASS')
    total = len(test_results['tests'])
    test_results['summary'] = f"{passed}/{total} tests passed"
    
    return test_results
