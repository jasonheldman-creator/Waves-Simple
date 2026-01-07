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

# Import price_book module at module level to avoid redundant imports
try:
    import price_book
    PRICE_BOOK_AVAILABLE = True
except ImportError:
    PRICE_BOOK_AVAILABLE = False
    price_book = None

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
        if not PRICE_BOOK_AVAILABLE:
            metadata['errors'].append("price_book module not available")
            return metadata
        
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
    
    This function:
    1. Loads fresh price_book from cache (prices_cache.parquet)
    2. Exports it to prices.csv format for build_wave_history_from_prices.py
    3. Runs build_wave_history_from_prices.py
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        prices_csv_path = os.path.join(os.getcwd(), 'prices.csv')
        
        # Step 1: Load fresh price_book from cache
        try:
            if not PRICE_BOOK_AVAILABLE:
                return False, "price_book module not available"
            
            price_data = price_book.get_price_book()
            
            if price_data.empty:
                return False, "Price cache is empty, cannot rebuild wave_history"
            
            price_book_max_date = price_data.index[-1].strftime('%Y-%m-%d')
            logger.info(f"Loaded price_book with max date: {price_book_max_date}")
            
        except Exception as e:
            return False, f"Failed to load price_book: {str(e)}"
        
        # Step 2: Export price_book to prices.csv format for build script
        try:
            # Convert price_book (wide format) to prices.csv (long format)
            # prices.csv format: date, ticker, close
            
            # Use pd.melt for efficient reshaping
            price_data_reset = price_data.reset_index()
            prices_long_df = pd.melt(
                price_data_reset,
                id_vars=['index'],
                var_name='ticker',
                value_name='close'
            )
            prices_long_df = prices_long_df.rename(columns={'index': 'date'})
            
            # Remove NaN values
            prices_long_df = prices_long_df.dropna(subset=['close'])
            
            # Format date column
            prices_long_df['date'] = pd.to_datetime(prices_long_df['date']).dt.strftime('%Y-%m-%d')
            
            # Save to CSV
            prices_long_df.to_csv(prices_csv_path, index=False)
            
            logger.info(f"Exported {len(prices_long_df)} price records to {prices_csv_path}")
            
        except Exception as e:
            return False, f"Failed to export price_book to prices.csv: {str(e)}"
        
        # Step 3: Use build_wave_history_from_prices.py script
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
                
                # Count rows and verify max date
                try:
                    wave_history = pd.read_csv(wave_history_path)
                    row_count = len(wave_history)
                    wave_count = wave_history['wave'].nunique() if 'wave' in wave_history.columns else 0
                    
                    # Verify max date matches price_book
                    if 'date' in wave_history.columns:
                        wave_history['date'] = pd.to_datetime(wave_history['date'])
                        wave_history_max_date = wave_history['date'].max().strftime('%Y-%m-%d')
                        
                        return True, f"✅ wave_history.csv rebuilt successfully!\n\nFile: {wave_history_path}\nSize: {size_kb:.1f} KB\nRows: {row_count:,}\nWaves: {wave_count}\nMax date: {wave_history_max_date}\nUpdated: {mod_time}"
                    else:
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


def force_ledger_recompute() -> Tuple[bool, str]:
    """
    Force ledger recompute by reloading price_book cache and rebuilding wave_history.
    
    This function:
    1. Reloads price_book from data/cache/prices_cache.parquet
    2. Rebuilds wave_history.csv from price_book
    3. Returns diagnostic info about the recompute
    
    The actual ledger recompute will happen when session state is cleared
    and the UI requests the ledger again.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        results = []
        
        # Step 1: Check if price_book cache exists and get its max date
        cache_path = os.path.join(os.getcwd(), 'data', 'cache', 'prices_cache.parquet')
        if not os.path.exists(cache_path):
            return False, f"Price cache not found at {cache_path}. Cannot proceed with recompute."
        
        try:
            import price_book
            # Force reload from cache by clearing any cached state
            price_data = price_book.get_price_book()
            
            if price_data.empty:
                return False, "Price cache loaded but is empty. Cannot proceed with recompute."
            
            price_book_max_date = price_data.index[-1].strftime('%Y-%m-%d')
            price_book_rows = len(price_data)
            price_book_tickers = len(price_data.columns)
            
            results.append(f"✅ Price cache loaded: {price_book_rows} days × {price_book_tickers} tickers")
            results.append(f"   Max date: {price_book_max_date}")
            
        except Exception as e:
            return False, f"Failed to load price_book: {str(e)}"
        
        # Step 2: Rebuild wave_history from price_book
        logger.info("Rebuilding wave_history from price_book...")
        success, message = rebuild_wave_history()
        
        if not success:
            return False, f"Price cache loaded but wave_history rebuild failed:\n{message}"
        
        results.append(f"✅ wave_history.csv rebuilt")
        
        # Step 3: Verify wave_history max date matches price_book
        try:
            wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
            if os.path.exists(wave_history_path):
                wave_history = pd.read_csv(wave_history_path)
                if 'date' in wave_history.columns and not wave_history.empty:
                    wave_history['date'] = pd.to_datetime(wave_history['date'])
                    wave_history_max_date = wave_history['date'].max().strftime('%Y-%m-%d')
                    
                    if wave_history_max_date == price_book_max_date:
                        results.append(f"✅ wave_history max date matches price_book: {wave_history_max_date}")
                    else:
                        results.append(f"⚠️ wave_history max date ({wave_history_max_date}) differs from price_book ({price_book_max_date})")
        except Exception as e:
            results.append(f"⚠️ Could not verify wave_history max date: {str(e)}")
        
        # Step 4: Return success with diagnostic info
        final_message = "\n".join(results)
        final_message += "\n\n💡 Ledger will recompute automatically when accessed."
        final_message += f"\n💡 Expected Ledger max date: {price_book_max_date}"
        
        return True, final_message
        
    except Exception as e:
        logger.error(f"Error in force_ledger_recompute: {e}", exc_info=True)
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
    
    # Test 5a: Check VIX execution state in wave_history.csv
    try:
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        if os.path.exists(wave_history_path):
            import pandas as pd
            df = pd.read_csv(wave_history_path)
            
            # Check for VIX execution state columns
            required_vix_cols = ['vix_level', 'vix_regime', 'exposure_used', 'overlay_active']
            has_vix_cols = all(col in df.columns for col in required_vix_cols)
            
            if has_vix_cols:
                # Check if any recent data has VIX state
                df['date'] = pd.to_datetime(df['date'])
                latest_date = df['date'].max()
                latest_data = df[df['date'] == latest_date]
                
                # Count equity waves with active VIX overlay
                active_vix = latest_data[
                    (latest_data['overlay_active'] == True) & 
                    (latest_data['vix_level'].notna())
                ]
                
                if len(active_vix) > 0:
                    test_results['tests'].append({
                        'name': 'VIX execution state LIVE',
                        'status': 'PASS',
                        'message': f'Found {len(active_vix)} waves with active VIX overlay for {latest_date.strftime("%Y-%m-%d")}'
                    })
                else:
                    test_results['tests'].append({
                        'name': 'VIX execution state LIVE',
                        'status': 'WARN',
                        'message': 'VIX columns exist but no active overlays for latest date'
                    })
            else:
                missing = [c for c in required_vix_cols if c not in df.columns]
                test_results['tests'].append({
                    'name': 'VIX execution state LIVE',
                    'status': 'WARN',
                    'message': f'wave_history.csv missing VIX columns: {missing}. Run rebuild_wave_history() to add them.'
                })
        else:
            test_results['tests'].append({
                'name': 'VIX execution state LIVE',
                'status': 'FAIL',
                'message': 'wave_history.csv not found'
            })
            test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'VIX execution state LIVE',
            'status': 'FAIL',
            'message': f'Error checking VIX state: {str(e)}'
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
        if not PRICE_BOOK_AVAILABLE:
            test_results['tests'].append({
                'name': 'Load price data',
                'status': 'FAIL',
                'message': 'price_book module not available'
            })
            test_results['overall_status'] = 'FAIL'
        else:
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
    
    # Test 9: Ledger recompute readiness (price_book max date validation)
    try:
        if not PRICE_BOOK_AVAILABLE:
            test_results['tests'].append({
                'name': 'Ledger recompute readiness',
                'status': 'FAIL',
                'message': 'price_book module not available'
            })
            test_results['overall_status'] = 'FAIL'
        else:
            price_data = price_book.get_price_book()
            
            if not price_data.empty:
                price_book_max_date = price_data.index[-1].strftime('%Y-%m-%d')
                
                # Check if wave_history exists and matches
                wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
                if os.path.exists(wave_history_path):
                    wave_history = pd.read_csv(wave_history_path)
                    if 'date' in wave_history.columns and not wave_history.empty:
                        wave_history['date'] = pd.to_datetime(wave_history['date'])
                        wave_history_max_date = wave_history['date'].max().strftime('%Y-%m-%d')
                        
                        if wave_history_max_date == price_book_max_date:
                            test_results['tests'].append({
                                'name': 'Ledger recompute readiness',
                                'status': 'PASS',
                                'message': f'price_book and wave_history aligned at {price_book_max_date}'
                            })
                        else:
                            test_results['tests'].append({
                                'name': 'Ledger recompute readiness',
                                'status': 'WARN',
                                'message': f'price_book ({price_book_max_date}) and wave_history ({wave_history_max_date}) dates differ. Run rebuild_wave_history() to sync.'
                            })
                    else:
                        test_results['tests'].append({
                            'name': 'Ledger recompute readiness',
                            'status': 'WARN',
                            'message': 'wave_history.csv is empty or missing date column'
                        })
                else:
                    test_results['tests'].append({
                        'name': 'Ledger recompute readiness',
                        'status': 'WARN',
                        'message': 'wave_history.csv not found. Run rebuild_wave_history() to create it.'
                    })
            else:
                test_results['tests'].append({
                    'name': 'Ledger recompute readiness',
                    'status': 'FAIL',
                    'message': 'price_book is empty'
                })
                test_results['overall_status'] = 'FAIL'
    except Exception as e:
        test_results['tests'].append({
            'name': 'Ledger recompute readiness',
            'status': 'FAIL',
            'message': f'Error: {str(e)}'
        })
        test_results['overall_status'] = 'FAIL'
    
    # Generate summary
    passed = sum(1 for t in test_results['tests'] if t['status'] == 'PASS')
    total = len(test_results['tests'])
    test_results['summary'] = f"{passed}/{total} tests passed"
    
    return test_results
