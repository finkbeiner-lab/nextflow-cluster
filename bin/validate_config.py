#!/usr/bin/env python3
"""Validation script for finkbeiner.config (Nextflow config).

This validator parses each non-comment line and checks that it has the
exact form::

    params.NAME = VALUE        // optional comment

where VALUE is a quoted string, a number, a boolean, ``null``, or a
balanced list.  Anything else — stray characters, broken quotes, smart
quotes, hidden Unicode, BOM, duplicate parameters, or comments using
``#`` instead of ``//`` — is flagged.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Character classes that catch invisible / look-alike issues
# ---------------------------------------------------------------------------
SMART_QUOTES = {
    '‘': "'",  # LEFT SINGLE QUOTATION MARK
    '’': "'",  # RIGHT SINGLE QUOTATION MARK
    '“': '"',  # LEFT DOUBLE QUOTATION MARK
    '”': '"',  # RIGHT DOUBLE QUOTATION MARK
}
INVISIBLE_CHARS = {
    ' ': 'NO-BREAK SPACE',
    '​': 'ZERO WIDTH SPACE',
    '‌': 'ZERO WIDTH NON-JOINER',
    '‍': 'ZERO WIDTH JOINER',
    '﻿': 'BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE',
}

PARAM_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def add(issues: List[dict], line_num: int, col: Optional[int],
        category: str, message: str, content: str) -> None:
    """Append a uniformly-shaped issue dict."""
    issues.append({
        'line': line_num,
        'col': col,
        'type': category,
        'message': message,
        'content': content,
    })


# ---------------------------------------------------------------------------
# Pre-parse checks (run on raw bytes / raw strings)
# ---------------------------------------------------------------------------

def check_encoding_and_bom(path: str, issues: List[dict]) -> Optional[str]:
    """Return decoded text or None on failure; record BOM/encoding issues."""
    with open(path, 'rb') as f:
        raw = f.read()

    if raw.startswith(b'\xef\xbb\xbf'):
        add(issues, 1, 1, 'bom',
            'File starts with a UTF-8 BOM (byte-order mark). Remove it.',
            repr(raw[:3]))
        raw = raw[3:]

    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError as e:
        # Find the offending byte's line number
        prefix = raw[:e.start]
        line_num = prefix.count(b'\n') + 1
        add(issues, line_num, None, 'non_utf8',
            f'File contains non-UTF-8 byte at offset {e.start}: {raw[e.start:e.start+1]!r}',
            repr(raw[max(0, e.start - 10):e.start + 10]))
        return raw.decode('utf-8', errors='replace')


def check_invisible_chars(lines: List[str], issues: List[dict]) -> None:
    """Flag smart quotes, NBSP, zero-width chars anywhere in the file."""
    for line_num, line in enumerate(lines, 1):
        for col, ch in enumerate(line, 1):
            if ch in SMART_QUOTES:
                add(issues, line_num, col, 'smart_quote',
                    f"Smart quote {ch!r} (U+{ord(ch):04X}) at column {col}. "
                    f"Replace with ASCII {SMART_QUOTES[ch]!r}.",
                    repr(line.rstrip('\n')))
            elif ch in INVISIBLE_CHARS:
                add(issues, line_num, col, 'invisible_char',
                    f"Invisible character at column {col}: {INVISIBLE_CHARS[ch]} (U+{ord(ch):04X}).",
                    repr(line.rstrip('\n')))


# ---------------------------------------------------------------------------
# Line-level structural checks
# ---------------------------------------------------------------------------

def strip_inline_comment(text: str) -> str:
    """Remove a trailing ``//...`` comment while respecting quotes.

    Walks the string char-by-char so that ``//`` inside a quoted value
    (e.g. ``'http://foo'``) is not treated as the start of a comment.
    """
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            i += 2  # skip escaped char
            continue
        if not in_double and ch == "'":
            in_single = not in_single
        elif not in_single and ch == '"':
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == '/' and i + 1 < len(text) and text[i + 1] == '/':
                return text[:i].rstrip()
        i += 1
    return text.rstrip()


def validate_value(value: str) -> Optional[str]:
    """Return None if *value* parses cleanly, else an error message.

    Accepts: quoted strings, integers/floats (incl. scientific), booleans,
    ``null``, or balanced lists ``[...]``.  Anything trailing the value
    (e.g. ``10x``, ``10 5``) is rejected.
    """
    v = value.strip()
    if not v:
        return "Value is empty"

    # Quoted string — must start and end with the same quote, no unescaped
    # quotes in between.
    if v[0] in ("'", '"'):
        quote = v[0]
        if len(v) < 2 or v[-1] != quote:
            return f"Quoted value not closed with matching {quote}"
        inner = v[1:-1]
        # Reject unescaped quote of the same type inside
        i = 0
        while i < len(inner):
            if inner[i] == '\\' and i + 1 < len(inner):
                i += 2
                continue
            if inner[i] == quote:
                return f"Unescaped {quote} inside quoted value at position {i+1}"
            i += 1
        return None

    # List literal
    if v.startswith('['):
        if not v.endswith(']'):
            return "List value is not closed with ']'"
        depth = 0
        in_single = False
        in_double = False
        i = 0
        while i < len(v):
            ch = v[i]
            if ch == '\\' and i + 1 < len(v):
                i += 2
                continue
            if not in_double and ch == "'":
                in_single = not in_single
            elif not in_single and ch == '"':
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0 and i != len(v) - 1:
                        return f"Stray characters after closing ']': {v[i+1:]!r}"
            i += 1
        if depth != 0:
            return f"Unbalanced brackets in list (depth={depth})"
        if in_single or in_double:
            return "Unclosed quote inside list value"
        return None

    # Boolean / null
    if v in ('true', 'false', 'null'):
        return None

    # Number (int, float, scientific)
    if re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', v):
        return None

    return (f"Unrecognised value {v!r}. Expected a quoted string, number, "
            f"boolean, null, or list.")


def check_lines(lines: List[str], issues: List[dict]) -> None:
    """Parse each non-comment, non-blank line as ``params.NAME = VALUE``."""
    seen_params: Dict[str, int] = {}
    in_block_comment = False

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip('\n').rstrip('\r')

        # Handle /* ... */ block comments (multi-line allowed)
        if in_block_comment:
            end = line.find('*/')
            if end >= 0:
                in_block_comment = False
                line = line[end + 2:]
            else:
                continue
        # Strip any embedded /* ... */ that opens and closes on this line
        while True:
            start = line.find('/*')
            if start < 0:
                break
            end = line.find('*/', start + 2)
            if end < 0:
                in_block_comment = True
                line = line[:start]
                break
            line = line[:start] + line[end + 2:]

        # Blank or // comment-only line
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            # Trailing whitespace check still applies
            if raw_line.rstrip('\n').rstrip('\r') != line.rstrip() and stripped:
                pass  # comment lines: trailing whitespace not a problem
            continue

        # Tabs vs spaces in leading whitespace — flag tabs (Nextflow style)
        leading = line[:len(line) - len(line.lstrip())]
        if '\t' in leading:
            add(issues, line_num, leading.index('\t') + 1, 'tab_indent',
                "Line uses tab indentation; use spaces.",
                repr(raw_line.rstrip('\n')))

        # Trailing whitespace on the code part
        if raw_line.rstrip('\n').rstrip('\r').rstrip(' \t') != raw_line.rstrip('\n').rstrip('\r'):
            add(issues, line_num, None, 'trailing_whitespace',
                "Line has trailing whitespace.",
                repr(raw_line.rstrip('\n')))

        # `#` comments are not valid in Nextflow config
        code_no_comment = strip_inline_comment(stripped)
        if code_no_comment.lstrip().startswith('#'):
            add(issues, line_num, None, 'hash_comment',
                "Use '//' for comments, not '#'.",
                repr(raw_line.rstrip('\n')))
            continue

        # Strip a trailing // comment for parsing
        code = strip_inline_comment(line)
        code_stripped = code.strip()
        if not code_stripped:
            continue

        # Must start with "params."
        if not code_stripped.startswith('params.'):
            add(issues, line_num, None, 'not_a_params_line',
                f"Line does not start with 'params.': {code_stripped!r}",
                repr(raw_line.rstrip('\n')))
            continue

        # Split on the first '=' that is NOT inside quotes
        eq_pos = _find_top_level_eq(code_stripped)
        if eq_pos is None:
            add(issues, line_num, None, 'missing_equals',
                "Parameter line has no '=' assignment.",
                repr(raw_line.rstrip('\n')))
            continue

        # Check for `==` typo
        if eq_pos + 1 < len(code_stripped) and code_stripped[eq_pos + 1] == '=':
            add(issues, line_num, eq_pos + 1, 'double_equals',
                "Use a single '=' for assignment, not '=='.",
                repr(raw_line.rstrip('\n')))
            continue

        lhs = code_stripped[:eq_pos].strip()
        rhs = code_stripped[eq_pos + 1:].strip()

        # LHS must be "params.NAME" — validate NAME
        if not lhs.startswith('params.'):
            add(issues, line_num, None, 'bad_lhs',
                f"Left-hand side must be 'params.NAME', got {lhs!r}",
                repr(raw_line.rstrip('\n')))
            continue
        name = lhs[len('params.'):]
        if not PARAM_NAME_RE.match(name):
            add(issues, line_num, None, 'bad_param_name',
                f"Invalid parameter name {name!r}. Use letters, digits, underscore; "
                f"cannot start with a digit.",
                repr(raw_line.rstrip('\n')))
            continue

        # Validate RHS
        err = validate_value(rhs)
        if err:
            add(issues, line_num, None, 'bad_value',
                f"params.{name}: {err}",
                repr(raw_line.rstrip('\n')))
            continue

        # Duplicate detection
        if name in seen_params:
            add(issues, line_num, None, 'duplicate_param',
                f"Parameter 'params.{name}' is defined again (first defined on line {seen_params[name]}).",
                repr(raw_line.rstrip('\n')))
        else:
            seen_params[name] = line_num


def _find_top_level_eq(s: str) -> Optional[int]:
    """Return the index of the first ``=`` that is NOT inside quotes/brackets."""
    in_single = False
    in_double = False
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            i += 2
            continue
        if not in_double and ch == "'":
            in_single = not in_single
        elif not in_single and ch == '"':
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
            elif ch == '=' and depth == 0:
                return i
        i += 1
    return None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def validate_config_file(config_path: str) -> int:
    """Run all checks. Return 0 if clean, 1 if any issues found."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    issues: List[dict] = []
    text = check_encoding_and_bom(config_path, issues)
    if text is None:
        text = ''

    # Preserve line numbering — splitlines(keepends=True) makes BOM-affected
    # cases align with the raw line indexing.
    lines = text.splitlines(keepends=False)

    check_invisible_chars(lines, issues)
    check_lines(lines, issues)

    if not issues:
        print(f"\n[OK] Config file validation passed: {config_path}\n")
        return 0

    print(f"\n{'=' * 80}")
    print(f"CONFIG VALIDATION REPORT: {config_path}")
    print(f"{'=' * 80}\n")

    issues.sort(key=lambda i: (i['line'], i.get('col') or 0))
    by_type: Dict[str, List[dict]] = {}
    for issue in issues:
        by_type.setdefault(issue['type'], []).append(issue)

    print(f"Found {len(issues)} issue(s) across {len(by_type)} categor{'y' if len(by_type) == 1 else 'ies'}:\n")
    for cat in sorted(by_type):
        cat_issues = by_type[cat]
        print(f"\n{cat.upper().replace('_', ' ')} ({len(cat_issues)} issue(s)):")
        print('-' * 80)
        for issue in cat_issues:
            loc = f"Line {issue['line']:4d}"
            if issue.get('col'):
                loc += f", col {issue['col']}"
            print(f"  {loc}: {issue['message']}")
            print(f"           Content: {issue['content']}")

    print(f"\n{'=' * 80}")
    print("Fix these issues before running the pipeline.")
    print(f"{'=' * 80}\n")
    return 1


def main() -> None:
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = str(Path.cwd() / 'finkbeiner.config')
    sys.exit(validate_config_file(config_path))


if __name__ == '__main__':
    main()
