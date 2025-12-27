"""
Startup Validation and Readiness Checks
Ensures critical systems are ready before rendering the main UI.
"""

import streamlit as st
from typing import Dict, List, Tuple
from datetime import datetime
import time


class ReadinessCheck:
    """Individual readiness check definition."""
    
    def __init__(self, name: str, check_func, critical: bool = True):
        """
        Initialize a readiness check.
        
        Args:
            name: Display name for the check
            check_func: Callable that returns (success: bool, message: str)
            critical: If True, failure blocks app startup
        """
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.status = "pending"
        self.message = ""
        self.duration_ms = 0
    
    def run(self) -> bool:
        """
        Execute the check.
        
        Returns:
            True if check passed, False otherwise
        """
        start = time.time()
        try:
            success, message = self.check_func()
            self.status = "passed" if success else "failed"
            self.message = message
            self.duration_ms = int((time.time() - start) * 1000)
            return success
        except Exception as e:
            self.status = "error"
            self.message = f"Exception: {str(e)}"
            self.duration_ms = int((time.time() - start) * 1000)
            return False


def check_data_files() -> Tuple[bool, str]:
    """Check if critical data files exist."""
    import os
    
    base_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(base_dir)
    
    critical_files = [
        'data/master_universe.csv',
        'wave_config.csv',
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = os.path.join(parent_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        return False, f"Missing files: {', '.join(missing_files)}"
    return True, "All critical data files present"


def check_imports() -> Tuple[bool, str]:
    """Check if critical Python packages are available."""
    missing_packages = []
    
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import plotly
    except ImportError:
        missing_packages.append("plotly")
    
    try:
        import yfinance
    except ImportError:
        # yfinance is important but not critical
        pass
    
    if missing_packages:
        return False, f"Missing packages: {', '.join(missing_packages)}"
    return True, "All critical packages available"


def check_helpers_available() -> Tuple[bool, str]:
    """Check if helper modules are available."""
    try:
        from helpers import ticker_rail, ticker_sources
        return True, "Helper modules loaded successfully"
    except ImportError as e:
        return False, f"Helper import failed: {str(e)}"


def check_waves_engine() -> Tuple[bool, str]:
    """Check if waves engine is available."""
    try:
        from waves_engine import get_all_waves
        waves = get_all_waves()
        if waves and len(waves) > 0:
            return True, f"Waves engine ready ({len(waves)} waves)"
        return False, "Waves engine returned no waves"
    except Exception as e:
        return False, f"Waves engine error: {str(e)}"


def check_resilience_features() -> Tuple[bool, str]:
    """Check if resilience features are available (non-critical)."""
    try:
        from helpers.circuit_breaker import get_circuit_breaker
        from helpers.persistent_cache import get_persistent_cache
        return True, "Resilience features active"
    except ImportError:
        return False, "Resilience features not available (non-critical)"


def run_startup_validation(show_progress: bool = True) -> Dict[str, any]:
    """
    Run all startup validation checks.
    
    Args:
        show_progress: If True, display progress in Streamlit
        
    Returns:
        Dict with validation results
    """
    checks = [
        ReadinessCheck("Data Files", check_data_files, critical=True),
        ReadinessCheck("Python Packages", check_imports, critical=True),
        ReadinessCheck("Helper Modules", check_helpers_available, critical=True),
        ReadinessCheck("Waves Engine", check_waves_engine, critical=True),
        ReadinessCheck("Resilience Features", check_resilience_features, critical=False),
    ]
    
    results = {
        'all_passed': True,
        'critical_failed': False,
        'checks': [],
        'total_duration_ms': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    if show_progress:
        st.info("🔍 Running startup validation checks...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    total_checks = len(checks)
    start_time = time.time()
    
    for idx, check in enumerate(checks):
        if show_progress:
            progress = (idx + 1) / total_checks
            progress_bar.progress(progress)
            status_text.text(f"Checking: {check.name}...")
        
        success = check.run()
        
        check_result = {
            'name': check.name,
            'status': check.status,
            'message': check.message,
            'duration_ms': check.duration_ms,
            'critical': check.critical
        }
        results['checks'].append(check_result)
        
        if not success:
            results['all_passed'] = False
            if check.critical:
                results['critical_failed'] = True
        
        if show_progress:
            if check.status == "passed":
                status_text.success(f"✅ {check.name}: {check.message}")
            elif check.critical:
                status_text.error(f"❌ {check.name}: {check.message}")
            else:
                status_text.warning(f"⚠️ {check.name}: {check.message}")
    
    results['total_duration_ms'] = int((time.time() - start_time) * 1000)
    
    if show_progress:
        progress_bar.empty()
        
        if results['critical_failed']:
            st.error("❌ Critical validation checks failed. Please resolve issues before proceeding.")
        elif results['all_passed']:
            st.success(f"✅ All validation checks passed ({results['total_duration_ms']}ms)")
        else:
            st.warning("⚠️ Some non-critical checks failed, but app can continue.")
    
    return results


def render_validation_results(results: Dict[str, any]):
    """
    Render detailed validation results.
    
    Args:
        results: Results from run_startup_validation
    """
    st.subheader("🔍 Startup Validation Results")
    
    # Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total = len(results['checks'])
        passed = sum(1 for c in results['checks'] if c['status'] == 'passed')
        st.metric("Total Checks", f"{passed}/{total}")
    
    with col2:
        if results['all_passed']:
            st.success("✅ All Passed")
        elif results['critical_failed']:
            st.error("❌ Critical Failed")
        else:
            st.warning("⚠️ Some Failed")
    
    with col3:
        st.metric("Duration", f"{results['total_duration_ms']}ms")
    
    # Detailed results
    st.markdown("---")
    st.markdown("### Check Details")
    
    for check in results['checks']:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.text(check['name'])
        
        with col2:
            if check['status'] == 'passed':
                st.success("✅ Passed")
            elif check['status'] == 'failed':
                if check['critical']:
                    st.error("❌ Failed")
                else:
                    st.warning("⚠️ Failed")
            else:
                st.error("❌ Error")
        
        with col3:
            st.caption(f"{check['duration_ms']}ms")
        
        if check['message']:
            st.caption(f"   ↳ {check['message']}")
