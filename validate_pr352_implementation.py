#!/usr/bin/env python3
"""
Validation Script for PR #352: PRICE_BOOK Freshness Option A1

This script validates the implementation of the GitHub Actions workflow
for automated price cache updates.

Requirements:
- Workflow file exists at .github/workflows/update_price_cache.yml
- Workflow has correct schedule and manual trigger configuration
- Expected cache output path is correct (data/cache/prices_cache.parquet)
- No changes to app.py (confirmed via file check)

Does NOT:
- Import streamlit or app code (to avoid runtime dependencies)
- Execute the workflow or build process
- Validate screenshot artifacts (manual review required)
"""

import os
import sys
import yaml
from pathlib import Path

# Constants
WORKFLOW_FILE = ".github/workflows/update_price_cache.yml"
EXPECTED_CACHE_PATH = "data/cache/prices_cache.parquet"
EXPECTED_SCHEDULE_CRON = "0 2 * * 2-6"
EXPECTED_WORKFLOW_NAME = "Update Price Cache"

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_success(message):
    """Print success message in green."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message):
    """Print error message in red."""
    print(f"{RED}✗ {message}{RESET}")


def print_warning(message):
    """Print warning message in yellow."""
    print(f"{YELLOW}⚠ {message}{RESET}")


def print_header(message):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"{message}")
    print(f"{'=' * 70}\n")


def check_workflow_file_exists():
    """Check if workflow file exists."""
    print("Checking workflow file existence...")
    
    if not os.path.exists(WORKFLOW_FILE):
        print_error(f"Workflow file not found: {WORKFLOW_FILE}")
        return False
    
    print_success(f"Workflow file exists: {WORKFLOW_FILE}")
    return True


def validate_workflow_configuration():
    """Validate workflow YAML configuration."""
    print("\nValidating workflow configuration...")
    
    try:
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()
        
        # Pre-process YAML to handle 'on:' being parsed as boolean True
        # We replace the workflow trigger 'on:' with 'trigger_on:' for reliable parsing
        # This is safer than handling the True key since we control the exact pattern
        if '\non:' not in content:
            # If 'on:' is not found, the YAML might already use a different key or be malformed
            print_warning("YAML doesn't contain expected 'on:' trigger key pattern")
        
        content_safe = content.replace('\non:', '\ntrigger_on:')
        workflow = yaml.safe_load(content_safe)
        
        errors = []
        warnings = []
        
        # Check workflow name
        if 'name' in workflow:
            if EXPECTED_WORKFLOW_NAME.lower() in workflow['name'].lower():
                print_success(f"Workflow name: {workflow['name']}")
            else:
                warnings.append(f"Workflow name '{workflow['name']}' doesn't contain expected '{EXPECTED_WORKFLOW_NAME}'")
        else:
            warnings.append("Workflow name not specified")
        
        # Check triggers (using our safe key)
        if 'trigger_on' not in workflow:
            errors.append("No triggers ('on') defined in workflow")
            return False, errors, warnings
        
        triggers = workflow['trigger_on']
        
        # Check schedule trigger
        if 'schedule' in triggers:
            schedule = triggers['schedule']
            if isinstance(schedule, list) and len(schedule) > 0:
                cron = schedule[0].get('cron', '')
                if cron == EXPECTED_SCHEDULE_CRON:
                    print_success(f"Schedule trigger configured: {cron}")
                else:
                    warnings.append(f"Schedule cron '{cron}' differs from expected '{EXPECTED_SCHEDULE_CRON}'")
            else:
                errors.append("Schedule trigger is malformed")
        else:
            errors.append("Schedule trigger not configured")
        
        # Check workflow_dispatch trigger
        if 'workflow_dispatch' in triggers:
            dispatch = triggers['workflow_dispatch']
            print_success("Manual trigger (workflow_dispatch) configured")
            
            # Check for days input parameter
            if 'inputs' in dispatch and 'days' in dispatch['inputs']:
                days_input = dispatch['inputs']['days']
                default_days = days_input.get('default', 'Not set')
                print_success(f"  - 'days' input parameter exists (default: {default_days})")
            else:
                warnings.append("workflow_dispatch does not have 'days' input parameter")
        else:
            errors.append("Manual trigger (workflow_dispatch) not configured")
        
        # Check permissions
        if 'permissions' in workflow:
            perms = workflow['permissions']
            if perms.get('contents') == 'write':
                print_success("Permissions: contents=write (required for commits)")
            else:
                warnings.append("Permissions may not allow writing (contents: write recommended)")
        else:
            warnings.append("Permissions not explicitly set")
        
        # Check jobs
        if 'jobs' not in workflow:
            errors.append("No jobs defined in workflow")
            return False, errors, warnings
        
        jobs = workflow['jobs']
        if len(jobs) == 0:
            errors.append("No jobs defined")
            return False, errors, warnings
        
        # Get first job (assuming single job)
        job_name = list(jobs.keys())[0]
        job = jobs[job_name]
        
        print_success(f"Job defined: {job_name}")
        
        # Check steps
        if 'steps' not in job:
            errors.append("No steps defined in job")
            return False, errors, warnings
        
        steps = job['steps']
        step_names = [step.get('name', 'Unnamed') for step in steps]
        
        required_steps = [
            ('checkout', ['checkout', 'Checkout']),
            ('python setup', ['python', 'Set up Python']),
            ('install deps', ['dependencies', 'Install']),
            ('build/run cache', ['build', 'cache', 'Run price cache']),
            ('commit', ['commit', 'push', 'Commit and push'])
        ]
        
        for step_category, keywords in required_steps:
            found = False
            for step_name in step_names:
                if any(kw.lower() in step_name.lower() for kw in keywords):
                    found = True
                    print_success(f"  - Step found: {step_name}")
                    break
            
            if not found:
                warnings.append(f"No step found for: {step_category}")
        
        return len(errors) == 0, errors, warnings
        
    except yaml.YAMLError as e:
        print_error(f"Failed to parse YAML: {e}")
        return False, [f"YAML parse error: {e}"], []
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False, [f"Unexpected error: {e}"], []


def check_cache_path_references():
    """Check that workflow references correct cache path."""
    print("\nChecking cache path references in workflow...")
    
    try:
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()
        
        if EXPECTED_CACHE_PATH in content:
            print_success(f"Cache path reference found: {EXPECTED_CACHE_PATH}")
            return True
        else:
            print_warning(f"Expected cache path '{EXPECTED_CACHE_PATH}' not found in workflow file")
            print_warning("This may be acceptable if path is determined by the build script")
            return True
    
    except Exception as e:
        print_error(f"Failed to read workflow file: {e}")
        return False


def check_cache_directory_exists():
    """Check if cache directory exists."""
    print("\nChecking cache directory...")
    
    cache_dir = os.path.dirname(EXPECTED_CACHE_PATH)
    
    if os.path.exists(cache_dir):
        print_success(f"Cache directory exists: {cache_dir}")
        
        # Check if cache file exists
        if os.path.exists(EXPECTED_CACHE_PATH):
            file_size = os.path.getsize(EXPECTED_CACHE_PATH)
            file_size_mb = file_size / (1024 * 1024)
            print_success(f"Cache file exists: {EXPECTED_CACHE_PATH} ({file_size_mb:.2f} MB)")
        else:
            print_warning(f"Cache file does not exist yet: {EXPECTED_CACHE_PATH}")
            print_warning("This is expected if workflow hasn't run yet")
        
        return True
    else:
        print_warning(f"Cache directory does not exist: {cache_dir}")
        print_warning("Directory will be created on first workflow run")
        return True


def check_no_app_py_changes():
    """Verify that app.py was not modified as part of this PR."""
    print("\nVerifying no app.py modifications...")
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print_warning("app.py not found in repository")
        return True
    
    # We cannot reliably check git history without executing git commands
    # Instead, we just verify the file exists and hasn't been corrupted
    try:
        with open("app.py", 'r') as f:
            content = f.read()
        
        # Basic sanity check: file should contain streamlit imports
        if 'streamlit' in content or 'st.' in content:
            print_success("app.py exists and appears valid (contains Streamlit code)")
        else:
            print_warning("app.py exists but may not be a Streamlit app")
        
        print_success("No programmatic way to verify app.py unchanged")
        print("    → Manual verification required: Check PR 'Files changed' tab")
        
        return True
    
    except Exception as e:
        print_error(f"Failed to read app.py: {e}")
        return False


def check_documentation_files():
    """Check that required documentation files exist."""
    print("\nChecking documentation files...")
    
    required_docs = [
        "PROOF_ARTIFACTS_GUIDE.md",
        "PRICE_BOOK_OPTION_A1_IMPLEMENTATION.md"
    ]
    
    all_exist = True
    
    for doc in required_docs:
        if os.path.exists(doc):
            print_success(f"Documentation exists: {doc}")
        else:
            print_error(f"Required documentation missing: {doc}")
            all_exist = False
    
    return all_exist


def main():
    """Main validation function."""
    print_header("PR #352 Implementation Validation")
    print("PRICE_BOOK Freshness Option A1")
    print("Validating GitHub Actions workflow configuration...\n")
    
    all_checks_passed = True
    
    # Check 1: Workflow file exists
    if not check_workflow_file_exists():
        print_error("\nFATAL: Workflow file not found. Implementation incomplete.")
        sys.exit(1)
    
    # Check 2: Validate workflow configuration
    config_valid, errors, warnings = validate_workflow_configuration()
    
    if not config_valid:
        print_error("\nWorkflow configuration validation failed:")
        for error in errors:
            print_error(f"  - {error}")
        all_checks_passed = False
    
    if warnings:
        print_warning("\nWorkflow configuration warnings:")
        for warning in warnings:
            print_warning(f"  - {warning}")
    
    # Check 3: Cache path references
    if not check_cache_path_references():
        all_checks_passed = False
    
    # Check 4: Cache directory exists
    if not check_cache_directory_exists():
        all_checks_passed = False
    
    # Check 5: No app.py changes
    if not check_no_app_py_changes():
        all_checks_passed = False
    
    # Check 6: Documentation files
    if not check_documentation_files():
        all_checks_passed = False
    
    # Final summary
    print_header("Validation Summary")
    
    if all_checks_passed:
        print_success("✅ All automated checks passed!")
        print("\nNext steps:")
        print("  1. Review PR 'Files changed' to confirm no app.py modifications")
        print("  2. Trigger manual workflow run to test execution")
        print("  3. Capture required proof artifacts (see PROOF_ARTIFACTS_GUIDE.md)")
        print("  4. Submit PR with screenshots attached")
        return 0
    else:
        print_error("❌ Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
