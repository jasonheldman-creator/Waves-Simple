#!/usr/bin/env python3
"""
Vercel Configuration Validator

This script validates the vercel.json configuration to ensure:
1. No deprecated properties are present
2. Redirect rules are correctly configured to prevent loops
3. Domain configuration is optimal for deployment
"""

import json
import sys
from pathlib import Path


def validate_vercel_config(config_path: Path) -> list[str]:
    """Validate vercel.json configuration and return list of issues."""
    issues = []
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return ["vercel.json not found in repository root"]
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in vercel.json: {e}"]
    
    # Check for deprecated properties
    deprecated_props = ["buildCommand", "rootDirectory", "installCommand", "outputDirectory"]
    for prop in deprecated_props:
        if prop in config:
            issues.append(
                f"⚠️  Deprecated property '{prop}' found. "
                f"Configure this in Vercel Dashboard → Project Settings → General instead."
            )
    
    # Validate redirect configuration
    if "redirects" in config:
        redirects = config["redirects"]
        if not isinstance(redirects, list):
            issues.append("❌ 'redirects' should be an array")
        else:
            # Check for potential redirect loops
            for idx, redirect in enumerate(redirects):
                if not isinstance(redirect, dict):
                    issues.append(f"❌ Redirect at index {idx} is not an object")
                    continue
                
                source = redirect.get("source", "")
                destination = redirect.get("destination", "")
                has_conditions = redirect.get("has", [])
                
                # Check for redirect loop potential
                if source and destination:
                    # If there's no "has" condition, could be problematic
                    if not has_conditions and "/:path*" in source:
                        issues.append(
                            f"⚠️  Redirect {idx}: Wildcard redirect without 'has' condition "
                            f"may cause redirect loops"
                        )
                    
                    # Validate host-based redirect
                    for condition in has_conditions:
                        if condition.get("type") == "host":
                            host_value = condition.get("value", "")
                            if host_value:
                                # Extract destination hostname
                                dest_url = destination.replace("https://", "").replace("http://", "")
                                dest_host = dest_url.split("/")[0] if "/" in dest_url else dest_url.split(":")[0]
                                
                                # Check if source and destination are the same
                                if host_value == dest_host:
                                    issues.append(
                                        f"❌ Redirect {idx}: Source host '{host_value}' is same as "
                                        f"destination host '{dest_host}' - this will cause a redirect loop!"
                                    )
                                # This is valid - different hosts (e.g., non-www to www)
                                else:
                                    print(f"✓ Redirect {idx}: {host_value} → {dest_host} (valid)")
    
    # Validate overall structure
    valid_top_level_keys = {
        "redirects", "rewrites", "headers", "cleanUrls", "trailingSlash",
        "crons", "regions", "functions", "env"
    }
    
    unknown_keys = set(config.keys()) - valid_top_level_keys
    if unknown_keys:
        issues.append(
            f"⚠️  Unknown top-level keys found: {', '.join(unknown_keys)}. "
            f"These may be ignored or deprecated."
        )
    
    return issues


def main():
    """Main validation function."""
    print("🔍 Validating vercel.json configuration...\n")
    
    # Find vercel.json in repository root
    repo_root = Path(__file__).parent
    config_path = repo_root / "vercel.json"
    
    issues = validate_vercel_config(config_path)
    
    if not issues:
        print("✅ vercel.json configuration is valid!")
        print("\n📋 Configuration Summary:")
        print("   - No deprecated properties found")
        print("   - Redirect rules are properly configured")
        print("   - No potential redirect loops detected")
        print("\n⚠️  Remember to configure in Vercel Dashboard:")
        print("   - Root Directory: site")
        print("   - Framework Preset: Next.js")
        print("   - See VERCEL_SETUP.md for complete instructions")
        return 0
    else:
        print("❌ Issues found in vercel.json:\n")
        for issue in issues:
            print(f"   {issue}")
        print("\n📖 See VERCEL_SETUP.md for configuration guidance")
        return 1


if __name__ == "__main__":
    sys.exit(main())
