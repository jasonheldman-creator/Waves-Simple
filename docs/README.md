# Documentation

This directory contains comprehensive documentation for validation, verification, and compliance purposes.

## Documentation Files

### `INSTITUTIONAL_READINESS_VERIFICATION.md`

**Purpose**: Audit-ready verification documentation for the Institutional Readiness tab implementation.

**Contents**:
- **Proof Summary Block**: Scope statement, validation categories table, status summary
- **Implementation Overview**: Function references with stable anchors
- **Validation Details**: Complete verification narrative for all 6 checks
- **Architectural Context**: Tab layouts, design principles
- **CI/CD Integration**: GitHub Actions workflow overview
- **Audit Trail**: Change history and verification artifacts
- **Compliance Checklist**: Acceptance criteria verification

**Target Audience**: Auditors, compliance officers, technical reviewers

### `VALIDATION_RUNBOOK.md`

**Purpose**: Operational guide for running validation scripts and interpreting results.

**Contents**:
- **Quick Start**: Running the validation script
- **Expected Output**: Success and failure examples
- **Validation Checks Explained**: Detailed explanation of each check
- **Exit Codes**: Meaning and required actions
- **Troubleshooting**: Common issues and resolutions
- **Advanced Usage**: CI/CD integration, debugging techniques
- **Maintenance**: When and how to update validation scripts

**Target Audience**: Developers, DevOps engineers, QA teams

## Key Features

### Resilient Line References

Documentation uses **stable anchors** instead of relying solely on line numbers:

- **Function Names**: `render_overview_clean_tab()`
- **Constants**: `CONFIDENCE_HIGH_COVERAGE_PCT`, `RISK_REGIME_VIX_LOW`, etc.
- **Section Headers**: "🏛️ Institutional Readiness", "System Diagnostics", etc.
- **Expander Labels**: `st.expander("🔧 System Diagnostics...", expanded=False)`

This approach ensures documentation remains accurate even when line numbers shift due to code changes.

### Proof Summary Block

Each verification document includes a **Proof Summary Block** containing:

1. **Scope Statement**: What was validated (validation-only, no functional changes)
2. **Validation Categories Table**: Mapping requirements to proof locations
3. **Status Summary**: Overall validation status and results

This provides auditors with a quick overview before diving into details.

## Documentation Standards

### Structure

All verification documents should follow this structure:

1. **Summary Section**: Proof block, scope, status
2. **Implementation Details**: What was verified, how, and where
3. **Validation Results**: Test output, proof of compliance
4. **Appendices**: Additional context, references

### Style Guidelines

- Use clear, professional language
- Include specific proof locations (file paths, line numbers, anchors)
- Provide complete validation output logs
- Mark status clearly (✓/✗, PASS/FAIL)
- Include timestamps and version information

### Maintenance

When implementation changes:

1. Update proof locations if anchors change
2. Re-run validation scripts and update output logs
3. Update status summaries
4. Increment version in change history

## Related Resources

- **Validation Scripts**: `../validation/`
- **Results Summary**: `../VALIDATION_RESULTS.md`
- **CI Workflows**: `../.github/workflows/`

## Purpose

This documentation serves multiple purposes:

1. **Compliance**: Provides audit trail for regulatory requirements
2. **Quality Assurance**: Demonstrates verification of implementation
3. **Knowledge Transfer**: Helps new team members understand validation process
4. **Troubleshooting**: Assists in diagnosing validation failures

## Contributing

When adding new documentation:

1. Follow the existing structure and style
2. Include proof summary block
3. Use stable anchors for code references
4. Provide complete validation output
5. Update this README with new document descriptions
