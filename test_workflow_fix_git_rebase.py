#!/usr/bin/env python3
"""
Test for the build_wave_history.yml workflow fix.

This test validates that the workflow has been fixed to:
1. Remove git pull --rebase logic
2. Linearize operations correctly
3. Guard commits properly
"""

import sys
import yaml


def test_workflow_no_git_pull_rebase():
    """Test that the workflow doesn't contain git pull --rebase."""
    print("\n" + "=" * 80)
    print("TEST: Workflow doesn't contain 'git pull --rebase'")
    print("=" * 80)
    
    with open('.github/workflows/build_wave_history.yml', 'r') as f:
        content = f.read()
    
    assert 'git pull --rebase' not in content, \
        "Workflow should not contain 'git pull --rebase'"
    
    print("✓ No 'git pull --rebase' found in workflow")
    print("=" * 80)
    print("✓ PASSED: Git pull --rebase removed\n")


def test_workflow_has_proper_step_order():
    """Test that the workflow has the correct step order."""
    print("\n" + "=" * 80)
    print("TEST: Workflow has proper step linearization")
    print("=" * 80)
    
    with open('.github/workflows/build_wave_history.yml', 'r') as f:
        workflow = yaml.safe_load(f)
    
    # Get the build job steps
    steps = workflow['jobs']['build']['steps']
    step_names = [step.get('name', '') for step in steps]
    
    print("Workflow steps:")
    for i, name in enumerate(step_names, 1):
        if name:
            print(f"  {i}. {name}")
    
    # Verify critical steps are in the correct order
    checkout_idx = next((i for i, name in enumerate(step_names) if 'Checkout' in name), None)
    build_idx = next((i for i, name in enumerate(step_names) if 'Build wave history' in name), None)
    validate_idx = next((i for i, name in enumerate(step_names) if 'Validate wave_history.csv' in name), None)
    commit_idx = next((i for i, name in enumerate(step_names) if 'Commit and push' in name), None)
    
    assert checkout_idx is not None, "Checkout step not found"
    assert build_idx is not None, "Build step not found"
    assert validate_idx is not None, "Validate step not found"
    assert commit_idx is not None, "Commit step not found"
    
    assert checkout_idx < build_idx, "Checkout must come before Build"
    assert build_idx < validate_idx, "Build must come before Validate"
    assert validate_idx < commit_idx, "Validate must come before Commit"
    
    print("\n✓ Steps are in correct order:")
    print(f"  Checkout ({checkout_idx}) → Build ({build_idx}) → Validate ({validate_idx}) → Commit ({commit_idx})")
    print("=" * 80)
    print("✓ PASSED: Proper step linearization\n")


def test_workflow_guards_commits():
    """Test that the workflow guards commits properly."""
    print("\n" + "=" * 80)
    print("TEST: Workflow guards commits when no changes exist")
    print("=" * 80)
    
    with open('.github/workflows/build_wave_history.yml', 'r') as f:
        content = f.read()
    
    # Check that the commit step checks for changes before committing
    assert 'git diff --staged --quiet' in content, \
        "Workflow should check for staged changes before committing"
    
    assert 'exit 0' in content, \
        "Workflow should exit cleanly when no changes exist"
    
    print("✓ Workflow checks for staged changes before committing")
    print("✓ Workflow exits cleanly when no changes exist")
    print("=" * 80)
    print("✓ PASSED: Commits are properly guarded\n")


def test_workflow_stages_files_first():
    """Test that the workflow stages files before checking for changes."""
    print("\n" + "=" * 80)
    print("TEST: Workflow stages files before checking for changes")
    print("=" * 80)
    
    with open('.github/workflows/build_wave_history.yml', 'r') as f:
        content = f.read()
    
    # Find the commit step content
    commit_step_start = content.find('- name: Commit and push updated wave history')
    assert commit_step_start > 0, "Commit step not found"
    
    commit_step = content[commit_step_start:commit_step_start + 2000]
    
    # Check that staging happens before diff check
    stage_pos = commit_step.find('git add wave_history.csv')
    diff_pos = commit_step.find('git diff --staged --quiet')
    
    assert stage_pos > 0, "git add not found in commit step"
    assert diff_pos > 0, "git diff --staged not found in commit step"
    assert stage_pos < diff_pos, "git add must come before git diff --staged"
    
    print("✓ Files are staged before checking for changes")
    print("=" * 80)
    print("✓ PASSED: Staging happens first\n")


if __name__ == '__main__':
    try:
        test_workflow_no_git_pull_rebase()
        test_workflow_has_proper_step_order()
        test_workflow_guards_commits()
        test_workflow_stages_files_first()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Git pull --rebase removed")
        print("  ✓ Steps properly linearized")
        print("  ✓ Commits properly guarded")
        print("  ✓ Files staged before checking for changes")
        print()
        
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
