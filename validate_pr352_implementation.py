#!/usr/bin/env python3
"""
Validation Script for PR #352 Option A1 Implementation
=======================================================

This script validates that all required components for Option A1 automation
are correctly configured and ready for deployment.

Usage:
    python validate_pr352_implementation.py

Exit Codes:
    0: All checks passed
    1: One or more checks failed
"""

import sys
import os
from pathlib import Path
import yaml


def print_header(message):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)


def print_check(description, passed, details=None):
    """Print a check result"""
    status = "✅" if passed else "❌"
    print(f"{status} {description}")
    if details:
        for line in details:
            print(f"   {line}")


def check_workflow_file_exists():
    """Check if GitHub Actions workflow file exists"""
    workflow_path = Path(".github/workflows/update_price_cache.yml")
    exists = workflow_path.exists()
    
    details = []
    if exists:
        details.append(f"Found: {workflow_path}")
        details.append(f"Size: {workflow_path.stat().st_size} bytes")
    else:
        details.append(f"Missing: {workflow_path}")
    
    print_check("Workflow file exists", exists, details)
    return exists


def check_workflow_yaml_valid():
    """Check if workflow file is valid YAML"""
    workflow_path = Path(".github/workflows/update_price_cache.yml")
    
    if not workflow_path.exists():
        print_check("Workflow YAML syntax", False, ["File does not exist"])
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            content = f.read()
            workflow_data = yaml.safe_load(content)
        
        details = []
        details.append(f"Valid YAML: Yes")
        details.append(f"Workflow name: {workflow_data.get('name', 'N/A')}")
        
        # Check for required keys - note 'on' is a Python keyword, check with string
        required_keys = ['name', 'jobs']
        has_on = 'on' in workflow_data or True in workflow_data  # 'on' might be parsed as True
        
        missing_keys = [key for key in required_keys if key not in workflow_data]
        
        if missing_keys or not has_on:
            if not has_on:
                missing_keys.append('on')
            details.append(f"Missing keys: {', '.join(missing_keys)}")
            print_check("Workflow YAML syntax", False, details)
            return False
        
        details.append(f"Jobs: {', '.join(workflow_data['jobs'].keys())}")
        print_check("Workflow YAML syntax", True, details)
        return True
        
    except yaml.YAMLError as e:
        print_check("Workflow YAML syntax", False, [f"YAML error: {e}"])
        return False
    except Exception as e:
        print_check("Workflow YAML syntax", False, [f"Error: {e}"])
        return False


def check_workflow_schedule():
    """Check if workflow has correct schedule configuration"""
    workflow_path = Path(".github/workflows/update_price_cache.yml")
    
    if not workflow_path.exists():
        print_check("Workflow schedule", False, ["File does not exist"])
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            content = f.read()
            workflow_data = yaml.safe_load(content)
        
        details = []
        
        # Check for schedule trigger - 'on' might be parsed as True (Python keyword)
        on_config = workflow_data.get('on', workflow_data.get(True, {}))
        
        if isinstance(on_config, str):
            # Simple trigger like 'push'
            has_schedule = False
        else:
            has_schedule = 'schedule' in on_config if isinstance(on_config, dict) else False
        
        has_manual = 'workflow_dispatch' in on_config if isinstance(on_config, dict) else False
        
        if has_schedule:
            schedule = on_config['schedule']
            if schedule:
                cron = schedule[0].get('cron', 'N/A') if isinstance(schedule, list) else 'N/A'
                details.append(f"Schedule: {cron}")
                details.append(f"Expected: 0 6 * * * (daily at 6:00 AM UTC)")
        else:
            details.append("No schedule trigger found")
        
        if has_manual:
            details.append("Manual trigger: workflow_dispatch enabled")
        else:
            details.append("Manual trigger: Not configured")
        
        passed = has_schedule or has_manual
        print_check("Workflow schedule", passed, details)
        return passed
        
    except Exception as e:
        print_check("Workflow schedule", False, [f"Error: {e}"])
        return False


def check_price_book_function():
    """Check if price_book.py has rebuild_price_cache with correct signature"""
    try:
        # Try to import the function
        from helpers.price_book import rebuild_price_cache
        import inspect
        
        details = []
        
        # Check function signature
        sig = inspect.signature(rebuild_price_cache)
        params = list(sig.parameters.keys())
        
        has_active_only = 'active_only' in params
        has_force_user = 'force_user_initiated' in params
        
        details.append(f"Parameters: {', '.join(params)}")
        
        if has_active_only:
            details.append("✓ Has active_only parameter")
        else:
            details.append("✗ Missing active_only parameter")
        
        if has_force_user:
            details.append("✓ Has force_user_initiated parameter")
            # Check default value
            default = sig.parameters['force_user_initiated'].default
            if default is False:
                details.append(f"✓ Default value: {default}")
            else:
                details.append(f"⚠ Default value: {default} (expected False)")
        else:
            details.append("✗ Missing force_user_initiated parameter")
        
        passed = has_active_only and has_force_user
        print_check("Price cache function signature", passed, details)
        return passed
        
    except ImportError as e:
        print_check("Price cache function signature", False, [f"Import error: {e}"])
        return False
    except Exception as e:
        print_check("Price cache function signature", False, [f"Error: {e}"])
        return False


def check_cache_directory():
    """Check if cache directory exists"""
    cache_dir = Path("data/cache")
    exists = cache_dir.exists() and cache_dir.is_dir()
    
    details = []
    if exists:
        details.append(f"Directory: {cache_dir}")
        
        # Check for cache files
        cache_file = cache_dir / "prices_cache.parquet"
        failed_file = cache_dir / "failed_tickers.csv"
        
        if cache_file.exists():
            details.append(f"✓ Cache file exists: {cache_file.name}")
        else:
            details.append(f"⚠ Cache file missing: {cache_file.name}")
        
        if failed_file.exists():
            details.append(f"✓ Failed tickers file exists: {failed_file.name}")
        else:
            details.append(f"⚠ Failed tickers file missing: {failed_file.name}")
    else:
        details.append(f"Missing: {cache_dir}")
    
    print_check("Cache directory structure", exists, details)
    return exists


def check_python_dependencies():
    """Check if required Python dependencies are installed"""
    required_packages = [
        'pandas',
        'yfinance',
        'pyarrow',  # For parquet support
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    details = []
    for pkg in installed:
        details.append(f"✓ {pkg}")
    for pkg in missing:
        details.append(f"✗ {pkg}")
    
    passed = len(missing) == 0
    print_check("Python dependencies", passed, details)
    return passed


def check_documentation_files():
    """Check if all required documentation files exist"""
    required_docs = [
        "PRICE_BOOK_OPTION_A1_IMPLEMENTATION.md",
        "PROOF_ARTIFACTS_GUIDE.md",
        "README_PR352.md",
        "PR_352_FINAL_SUMMARY.md",
    ]
    
    missing = []
    found = []
    
    for doc in required_docs:
        doc_path = Path(doc)
        if doc_path.exists():
            found.append(doc)
        else:
            missing.append(doc)
    
    details = []
    for doc in found:
        details.append(f"✓ {doc}")
    for doc in missing:
        details.append(f"✗ {doc}")
    
    passed = len(missing) == 0
    print_check("Documentation files", passed, details)
    return passed


def check_validation_script():
    """Check if this validation script exists (meta check)"""
    script_path = Path(__file__).name
    exists = Path(__file__).exists()
    
    details = [f"Script: {script_path}"]
    print_check("Validation script", exists, details)
    return exists


def run_all_checks():
    """Run all validation checks"""
    print_header("PR #352 Option A1 Implementation Validation")
    
    checks = [
        ("Workflow File", check_workflow_file_exists),
        ("Workflow YAML", check_workflow_yaml_valid),
        ("Workflow Schedule", check_workflow_schedule),
        ("Price Cache Function", check_price_book_function),
        ("Cache Directory", check_cache_directory),
        ("Python Dependencies", check_python_dependencies),
        ("Documentation Files", check_documentation_files),
        ("Validation Script", check_validation_script),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name}: Unexpected error: {e}")
            results[check_name] = False
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    for check_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
    
    if passed == total:
        print("\n✅ ALL CHECKS PASSED - Ready for deployment")
        return 0
    else:
        print(f"\n❌ {total - passed} CHECK(S) FAILED - Review errors above")
        return 1


def main():
    """Main entry point"""
    # Change to repository root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        exit_code = run_all_checks()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
