"""
Wave Registry Validator

This module validates the wave registry configuration file and ensures:
- Exactly 28 enabled unique wave_ids
- No duplicate wave_ids or display_names
- Each wave has a benchmark definition
- Each wave has weights defined or a fallback mechanism

The validator runs on app start and provides detailed diagnostics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ValidationResult:
    """Result of a wave registry validation check."""
    is_valid: bool
    error_count: int
    warning_count: int
    errors: List[str]
    warnings: List[str]
    info: List[str]
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.error_count += 1
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        self.warning_count += 1
    
    def add_info(self, message: str):
        """Add an info message."""
        self.info.append(message)
    
    def get_summary(self) -> str:
        """Get a summary of the validation results."""
        if self.is_valid:
            return f"✓ Valid ({len(self.info)} info, {self.warning_count} warnings)"
        else:
            return f"✗ Invalid ({self.error_count} errors, {self.warning_count} warnings)"
    
    def get_detailed_report(self) -> str:
        """Get a detailed report of all validation messages."""
        lines = [f"Wave Registry Validation Report", "=" * 50, ""]
        
        if self.errors:
            lines.append("ERRORS:")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"  {i}. {err}")
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warn}")
            lines.append("")
        
        if self.info:
            lines.append("INFO:")
            for i, inf in enumerate(self.info, 1):
                lines.append(f"  {i}. {inf}")
            lines.append("")
        
        lines.append(f"Summary: {self.get_summary()}")
        return "\n".join(lines)


def load_wave_registry(registry_path: str = "config/wave_registry.json") -> Optional[Dict]:
    """
    Load the wave registry from file.
    
    Args:
        registry_path: Path to the wave registry JSON file
        
    Returns:
        Registry dictionary if successful, None otherwise
    """
    try:
        if not os.path.exists(registry_path):
            return None
        
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        return registry
    except Exception as e:
        return None


def validate_wave_registry(
    registry_path: str = "config/wave_registry.json",
    wave_weights: Optional[Dict] = None
) -> ValidationResult:
    """
    Validate the wave registry configuration.
    
    Checks:
    - Registry file exists and is valid JSON
    - Exactly 28 enabled unique wave_ids
    - No duplicate wave_ids or display_names
    - Each wave has a benchmark definition
    - Each wave has weights defined (if wave_weights provided)
    
    Args:
        registry_path: Path to the wave registry JSON file
        wave_weights: Optional WAVE_WEIGHTS dict from waves_engine for cross-validation
        
    Returns:
        ValidationResult with detailed diagnostics
    """
    result = ValidationResult(
        is_valid=True,
        error_count=0,
        warning_count=0,
        errors=[],
        warnings=[],
        info=[]
    )
    
    # Check if registry file exists
    if not os.path.exists(registry_path):
        result.add_error(f"Wave registry file not found: {registry_path}")
        return result
    
    # Load registry
    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in registry file: {str(e)}")
        return result
    except Exception as e:
        result.add_error(f"Failed to load registry file: {str(e)}")
        return result
    
    # Validate registry structure
    if "waves" not in registry:
        result.add_error("Registry missing 'waves' key")
        return result
    
    waves = registry.get("waves", [])
    
    if not isinstance(waves, list):
        result.add_error("Registry 'waves' must be a list")
        return result
    
    # Track unique values
    wave_ids_seen: Set[str] = set()
    display_names_seen: Set[str] = set()
    enabled_count = 0
    
    # Validate each wave entry
    for i, wave in enumerate(waves):
        wave_num = i + 1
        
        # Check required fields
        required_fields = ["wave_id", "display_name", "category", "benchmark_ticker", "mode", "beta_target", "enabled"]
        for field in required_fields:
            if field not in wave:
                result.add_error(f"Wave #{wave_num} missing required field: {field}")
                continue
        
        # Skip further checks if required fields missing
        if any(field not in wave for field in required_fields):
            continue
        
        wave_id = wave["wave_id"]
        display_name = wave["display_name"]
        enabled = wave["enabled"]
        
        # Check for duplicates
        if wave_id in wave_ids_seen:
            result.add_error(f"Duplicate wave_id: {wave_id}")
        else:
            wave_ids_seen.add(wave_id)
        
        if display_name in display_names_seen:
            result.add_error(f"Duplicate display_name: {display_name}")
        else:
            display_names_seen.add(display_name)
        
        # Count enabled waves
        if enabled:
            enabled_count += 1
        
        # Validate benchmark definition
        benchmark = wave.get("benchmark_ticker", "")
        if not benchmark or not isinstance(benchmark, str) or len(benchmark.strip()) == 0:
            result.add_error(f"Wave '{display_name}' missing valid benchmark_ticker")
        
        # Validate mode
        mode = wave.get("mode", "")
        valid_modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
        if mode not in valid_modes:
            result.add_warning(f"Wave '{display_name}' has non-standard mode: {mode} (valid: {valid_modes})")
        
        # Validate category
        category = wave.get("category", "")
        valid_categories = ["Equity", "Crypto", "Fixed Income", "Commodity", "Multi-Asset"]
        if category not in valid_categories:
            result.add_warning(f"Wave '{display_name}' has non-standard category: {category}")
        
        # Validate beta_target
        beta_target = wave.get("beta_target")
        if not isinstance(beta_target, (int, float)) or beta_target < 0 or beta_target > 2:
            result.add_warning(f"Wave '{display_name}' has unusual beta_target: {beta_target} (expected 0-2)")
        
        # Validate tag
        tag = wave.get("tag", "")
        valid_tags = ["LIVE", "SANDBOX", "DEPRECATED"]
        if tag not in valid_tags:
            result.add_warning(f"Wave '{display_name}' has non-standard tag: {tag} (valid: {valid_tags})")
    
    # Check total enabled wave count
    if enabled_count != 28:
        result.add_error(f"Expected exactly 28 enabled waves, found {enabled_count}")
    else:
        result.add_info(f"✓ Exactly 28 enabled waves")
    
    # Check total wave count
    result.add_info(f"Total waves in registry: {len(waves)}")
    result.add_info(f"Unique wave_ids: {len(wave_ids_seen)}")
    result.add_info(f"Unique display_names: {len(display_names_seen)}")
    
    # Cross-validate with WAVE_WEIGHTS if provided
    if wave_weights is not None:
        result.add_info(f"Cross-validating with WAVE_WEIGHTS...")
        
        # Check that all enabled waves have weights
        registry_display_names = {w["display_name"] for w in waves if w.get("enabled", False)}
        wave_weights_names = set(wave_weights.keys())
        
        # Waves in registry but not in WAVE_WEIGHTS
        missing_weights = registry_display_names - wave_weights_names
        if missing_weights:
            result.add_error(f"Waves in registry but missing from WAVE_WEIGHTS: {missing_weights}")
        
        # Waves in WAVE_WEIGHTS but not in registry
        extra_weights = wave_weights_names - registry_display_names
        if extra_weights:
            result.add_warning(f"Waves in WAVE_WEIGHTS but not enabled in registry: {extra_weights}")
        
        # Check that each wave has at least one holding
        for display_name in registry_display_names:
            if display_name in wave_weights:
                holdings = wave_weights[display_name]
                if not holdings or len(holdings) == 0:
                    result.add_error(f"Wave '{display_name}' has no holdings in WAVE_WEIGHTS")
                else:
                    result.add_info(f"✓ Wave '{display_name}' has {len(holdings)} holdings")
    
    return result


def get_wave_registry_config(registry_path: str = "config/wave_registry.json") -> Dict[str, Dict]:
    """
    Get wave configuration as a dictionary indexed by wave_id.
    
    Args:
        registry_path: Path to the wave registry JSON file
        
    Returns:
        Dictionary mapping wave_id to wave config
    """
    registry = load_wave_registry(registry_path)
    if not registry or "waves" not in registry:
        return {}
    
    config = {}
    for wave in registry.get("waves", []):
        if "wave_id" in wave:
            config[wave["wave_id"]] = wave
    
    return config


def get_enabled_waves(registry_path: str = "config/wave_registry.json") -> List[Dict]:
    """
    Get list of enabled waves from registry.
    
    Args:
        registry_path: Path to the wave registry JSON file
        
    Returns:
        List of enabled wave configurations
    """
    registry = load_wave_registry(registry_path)
    if not registry or "waves" not in registry:
        return []
    
    return [w for w in registry.get("waves", []) if w.get("enabled", False)]


if __name__ == "__main__":
    # Test validation
    result = validate_wave_registry()
    print(result.get_detailed_report())
