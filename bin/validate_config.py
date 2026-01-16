#!/usr/bin/env python3
"""
Validation script for finkbeiner.config file.
Checks for whitespace issues, unexpected backslashes, and other common configuration errors.
"""

import sys
import re
import os
from pathlib import Path


def check_trailing_whitespace(lines, filename):
    """Check for trailing whitespace on lines (spaces/tabs before newline)."""
    issues = []
    for line_num, line in enumerate(lines, 1):
        # Check for trailing spaces or tabs (not just the newline character)
        stripped = line.rstrip('\n\r')
        if stripped != line.rstrip() or (stripped and stripped[-1] in ' \t'):
            # Only flag if there are actual spaces/tabs before the newline
            if stripped.endswith(' ') or stripped.endswith('\t'):
                issues.append({
                    'line': line_num,
                    'type': 'trailing_whitespace',
                    'message': f"Line {line_num} has trailing whitespace (spaces or tabs)",
                    'content': repr(line)
                })
    return issues


def check_unexpected_backslashes(lines, filename):
    """Check for unexpected backslashes that might cause issues."""
    issues = []
    for line_num, line in enumerate(lines, 1):
        # Skip comment-only lines
        stripped = line.strip()
        if stripped.startswith('//') or not stripped:
            continue
        
        # Check for backslashes that aren't part of valid paths or escape sequences
        # In Nextflow config, backslashes in paths are typically not needed on Linux
        # But we should flag standalone backslashes that might be accidental
        if '\\' in line:
            # Check if it's a valid escape sequence or path
            # Valid cases: \n, \t, \r, \\, or part of a Windows path (though this is Linux)
            # Invalid: standalone \ at end of line, or \ followed by unexpected char
            
            # Check for standalone backslash at end of line (before comment)
            if line.rstrip().endswith('\\') and not line.rstrip().endswith('\\\\'):
                issues.append({
                    'line': line_num,
                    'type': 'unexpected_backslash',
                    'message': f"Line {line_num} ends with a backslash (might be accidental line continuation)",
                    'content': repr(line)
                })
            
            # Check for backslash not in quotes (outside string context)
            # This is tricky - we'll look for backslashes that aren't in quoted strings
            # Simple check: backslash not followed by n, t, r, or another backslash
            matches = list(re.finditer(r'\\(?![ntr\\])', line))
            for match in matches:
                # Check if we're inside quotes
                before_match = line[:match.start()]
                single_quotes = before_match.count("'") - before_match.count("\\'")
                double_quotes = before_match.count('"') - before_match.count('\\"')
                
                # If we're inside quotes, it might be okay (could be path)
                # But flag it anyway as potentially problematic on Linux
                if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                    # Not inside quotes - this is definitely suspicious
                    issues.append({
                        'line': line_num,
                        'type': 'unexpected_backslash',
                        'message': f"Line {line_num} contains a backslash outside of quotes (column {match.start() + 1})",
                        'content': repr(line)
                    })
                else:
                    # Inside quotes - might be Windows path, flag as warning
                    issues.append({
                        'line': line_num,
                        'type': 'backslash_in_path',
                        'message': f"Line {line_num} contains backslash in quoted string (might be Windows path on Linux system)",
                        'content': repr(line)
                    })
    
    return issues


def check_whitespace_in_values(lines, filename):
    """Check for problematic whitespace in parameter values."""
    issues = []
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
        
        # Check for parameter assignments
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                param_value = parts[1]
                
                # Check for leading/trailing whitespace in quoted values
                # This looks for patterns like: param = ' value ' or param = " value "
                quoted_value_match = re.search(r"['\"]([^'\"]*)['\"]", param_value)
                if quoted_value_match:
                    quoted_content = quoted_value_match.group(1)
                    if quoted_content.startswith(' ') or quoted_content.endswith(' '):
                        issues.append({
                            'line': line_num,
                            'type': 'whitespace_in_quoted_value',
                            'message': f"Line {line_num}: Parameter '{param_name}' has leading/trailing whitespace in quoted value",
                            'content': repr(line)
                        })
                
                # Check for tabs in values (might cause issues)
                if '\t' in param_value:
                    issues.append({
                        'line': line_num,
                        'type': 'tab_in_value',
                        'message': f"Line {line_num}: Parameter '{param_name}' contains tab character",
                        'content': repr(line)
                    })
    
    return issues


def check_malformed_assignments(lines, filename):
    """Check for malformed parameter assignments."""
    issues = []
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
        
        # Check for parameter assignments (only in non-comment part)
        # Split on comment marker to ignore equals in comments
        if '//' in line:
            code_part = line.split('//')[0]
        else:
            code_part = line
        
        if '=' in code_part:
            # Check for multiple equals signs in the code part (might be accidental)
            if code_part.count('=') > 1:
                issues.append({
                    'line': line_num,
                    'type': 'multiple_equals',
                    'message': f"Line {line_num} contains multiple '=' signs in parameter assignment",
                    'content': repr(line)
                })
    
    return issues


def check_empty_lines_with_whitespace(lines, filename):
    """Check for lines that appear empty but contain whitespace."""
    issues = []
    for line_num, line in enumerate(lines, 1):
        if line.strip() == '' and line != '\n' and line != '':
            issues.append({
                'line': line_num,
                'type': 'empty_line_with_whitespace',
                'message': f"Line {line_num} appears empty but contains whitespace",
                'content': repr(line)
            })
    return issues


def validate_config_file(config_path):
    """Main validation function."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1
    
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_trailing_whitespace(lines, config_path))
    all_issues.extend(check_unexpected_backslashes(lines, config_path))
    all_issues.extend(check_whitespace_in_values(lines, config_path))
    all_issues.extend(check_malformed_assignments(lines, config_path))
    all_issues.extend(check_empty_lines_with_whitespace(lines, config_path))
    
    # Report issues
    if all_issues:
        print(f"\n{'='*80}")
        print(f"CONFIG VALIDATION REPORT: {config_path}")
        print(f"{'='*80}\n")
        
        # Group issues by type
        issues_by_type = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Print summary
        print(f"Found {len(all_issues)} issue(s) across {len(issues_by_type)} category(ies):\n")
        
        for issue_type, type_issues in sorted(issues_by_type.items()):
            print(f"\n{issue_type.upper().replace('_', ' ')} ({len(type_issues)} issue(s)):")
            print("-" * 80)
            for issue in type_issues:
                print(f"  Line {issue['line']:4d}: {issue['message']}")
                print(f"           Content: {issue['content']}")
        
        print(f"\n{'='*80}")
        print("Please review and fix these issues before running your experiment.")
        print(f"{'='*80}\n")
        return 1
    else:
        print(f"\n✓ Config file validation passed: {config_path}")
        print("  No whitespace or backslash issues detected.\n")
        return 0


def main():
    """Main entry point."""
    # Default to finkbeiner.config in the parent directory
    script_dir = Path(__file__).parent
    default_config = script_dir.parent / 'finkbeiner.config'
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = str(default_config)
    
    exit_code = validate_config_file(config_path)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
