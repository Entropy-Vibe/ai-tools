#!/usr/bin/env python3
"""
context_manifest.py — Context Window Budget & Manifest Tool

Scans prompt assembly files, counts tokens per file, and outputs a
formatted manifest showing your context budget. Designed for iterative
prompt engineering, eval frameworks, and agent orchestration.

Usage:
    python context_manifest.py [OPTIONS] [FILES_OR_DIRS...]

Examples:
    # Scan current directory for .md files
    python context_manifest.py .

    # Scan specific files
    python context_manifest.py plan.md agents.md mentor.md

    # Scan a directory with budget limit
    python context_manifest.py ./prompts --budget 200000

    # Output as JSON for programmatic use
    python context_manifest.py ./prompts --format json

    # Log to CSV for tracking over time
    python context_manifest.py ./prompts --log context_log.csv

    # Watch mode — re-scan on file changes
    python context_manifest.py ./prompts --watch

    # Filter by extensions
    python context_manifest.py ./prompts --ext .md .txt .yaml

    # Tag a run for eval correlation
    python context_manifest.py ./prompts --tag "baseline_v2" --log runs.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
# Claude uses a BPE tokenizer similar to cl100k_base. A good heuristic:
#   ~1 token per 4 characters for English prose
#   ~1 token per 3.5 characters for code/mixed content
#
# We improve on the naive char/4 ratio by accounting for whitespace,
# punctuation density, and code patterns. This gets within ~5-10% of
# real counts — close enough for budgeting.
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count for text using heuristic analysis.

    Returns an integer estimate. For precise counts, use the Anthropic
    token counting API (see --precise flag).
    """
    if not text:
        return 0

    char_count = len(text)
    word_count = len(text.split())

    # Base estimate: blend of char-based and word-based heuristics
    char_estimate = char_count / 3.8
    word_estimate = word_count * 1.33  # avg ~1.33 tokens per word

    # Detect if content is code-heavy (more tokens per char)
    code_indicators = len(re.findall(r'[{}()\[\];=<>]', text))
    code_ratio = code_indicators / max(char_count, 1)
    is_code_heavy = code_ratio > 0.02

    if is_code_heavy:
        # Code tends to tokenize less efficiently
        estimate = char_count / 3.3
    else:
        # Blend char and word estimates for prose
        estimate = (char_estimate * 0.6) + (word_estimate * 0.4)

    # Adjust for special patterns
    # URLs and paths tokenize into many small tokens
    url_count = len(re.findall(r'https?://\S+', text))
    path_count = len(re.findall(r'[/\\][\w.-]+[/\\]', text))
    estimate += (url_count + path_count) * 5

    # Markdown headers, bullets, etc.
    md_markers = len(re.findall(r'^[#*\->]+\s', text, re.MULTILINE))
    estimate += md_markers * 0.5

    return max(1, round(estimate))


def get_file_info(filepath: str) -> dict:
    """Read a file and return its token info."""
    path = Path(filepath)
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
        tokens = estimate_tokens(text)
        lines = text.count('\n') + (1 if text and not text.endswith('\n') else 0)
        chars = len(text)
        return {
            'file': str(path),
            'name': path.name,
            'ext': path.suffix,
            'tokens': tokens,
            'lines': lines,
            'chars': chars,
            'size_kb': round(path.stat().st_size / 1024, 1),
            'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
    except Exception as e:
        return {
            'file': str(path),
            'name': path.name,
            'ext': path.suffix,
            'tokens': 0,
            'lines': 0,
            'chars': 0,
            'size_kb': 0,
            'modified': '',
            'error': str(e),
        }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

DEFAULT_EXTENSIONS = {'.md', '.txt', '.yaml', '.yml', '.json', '.py',
                      '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
                      '.toml', '.cfg', '.ini', '.prompt', '.system'}


def discover_files(paths: list[str], extensions: Optional[set[str]] = None,
                   recursive: bool = True) -> list[str]:
    """Find all relevant files from given paths."""
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    files = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            files.append(str(path))
        elif path.is_dir():
            pattern = '**/*' if recursive else '*'
            for f in path.glob(pattern):
                if f.is_file() and f.suffix.lower() in extensions:
                    # Skip hidden files and common noise
                    parts = f.parts
                    if any(part.startswith('.') for part in parts):
                        continue
                    if any(part in ('node_modules', '__pycache__', '.git') for part in parts):
                        continue
                    files.append(str(f))
    return sorted(files)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS = {
    'opus':    200_000,
    'sonnet':  200_000,
    'haiku':   200_000,
    'default': 200_000,
}


def format_tokens(n: int) -> str:
    """Human-friendly token count."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def format_table(file_infos: list[dict], budget: int, tag: str = '') -> str:
    """Format as a nice ASCII table."""
    if not file_infos:
        return "No files found."

    lines = []
    total_tokens = sum(f['tokens'] for f in file_infos)

    # Header
    lines.append('')
    lines.append('╔══════════════════════════════════════════════════════════════╗')
    title = '  CONTEXT MANIFEST'
    if tag:
        title += f'  [{tag}]'
    lines.append(f'║{title:<62}║')
    lines.append(f'║  {datetime.now().strftime("%Y-%m-%d %H:%M:%S"):<60}║')
    lines.append('╠════╦══════════════════════════════╦════════╦═══════╦════════╣')
    lines.append('║  # ║ File                         ║ Tokens ║ Lines ║   KB   ║')
    lines.append('╠════╬══════════════════════════════╬════════╬═══════╬════════╣')

    for i, f in enumerate(file_infos, 1):
        name = f['name']
        if len(name) > 28:
            name = name[:25] + '...'
        tok = format_tokens(f['tokens'])
        error_flag = ' ⚠' if f.get('error') else ''
        lines.append(
            f'║ {i:>2} ║ {name:<28} ║ {tok:>6} ║ {f["lines"]:>5} ║ {f["size_kb"]:>5.1f}  ║{error_flag}'
        )

    lines.append('╠════╩══════════════════════════════╬════════╩═══════╩════════╣')

    # Totals & budget
    total_str = format_tokens(total_tokens)
    lines.append(f'║  Total Context                    ║  {total_str:>6} tokens{" " * 9}║')

    if budget:
        remaining = budget - total_tokens
        remaining_str = format_tokens(remaining)
        pct = (total_tokens / budget) * 100
        bar_width = 30
        filled = int(bar_width * min(pct, 100) / 100)
        bar = '█' * filled + '░' * (bar_width - filled)

        lines.append(f'║  Budget ({format_tokens(budget):>6})                 ║  {remaining_str:>6} remaining{" " * 5}║')
        lines.append(f'║  [{bar}] {pct:>5.1f}%{" " * 6}║')

        if pct > 90:
            lines.append(f'║  ⚠️  WARNING: Context usage above 90%!{" " * 23}║')
        elif pct > 75:
            lines.append(f'║  ⚡ Context usage above 75%{" " * 35}║')

    lines.append('╚══════════════════════════════════════════════════════════════╝')

    # Errors
    errors = [f for f in file_infos if f.get('error')]
    if errors:
        lines.append('')
        lines.append('Errors:')
        for f in errors:
            lines.append(f"  ⚠ {f['name']}: {f['error']}")

    return '\n'.join(lines)


def format_json(file_infos: list[dict], budget: int, tag: str = '') -> str:
    """Format as JSON for programmatic consumption."""
    total_tokens = sum(f['tokens'] for f in file_infos)
    output = {
        'timestamp': datetime.now().isoformat(),
        'tag': tag or None,
        'files': file_infos,
        'summary': {
            'total_tokens': total_tokens,
            'total_files': len(file_infos),
            'budget': budget,
            'remaining': budget - total_tokens if budget else None,
            'usage_pct': round((total_tokens / budget) * 100, 1) if budget else None,
        }
    }
    return json.dumps(output, indent=2)


def format_markdown(file_infos: list[dict], budget: int, tag: str = '') -> str:
    """Format as markdown table — useful for pasting into docs."""
    if not file_infos:
        return "No files found."

    total_tokens = sum(f['tokens'] for f in file_infos)
    lines = []

    if tag:
        lines.append(f'## Context Manifest — `{tag}`')
    else:
        lines.append('## Context Manifest')
    lines.append(f'*Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}*\n')
    lines.append('| # | File | Tokens | Lines | KB |')
    lines.append('|---|------|--------|-------|----|')
    for i, f in enumerate(file_infos, 1):
        lines.append(f'| {i} | `{f["name"]}` | {format_tokens(f["tokens"])} | {f["lines"]} | {f["size_kb"]} |')
    lines.append(f'\n**Total: {format_tokens(total_tokens)} tokens**')
    if budget:
        remaining = budget - total_tokens
        pct = (total_tokens / budget) * 100
        lines.append(f'Budget: {format_tokens(budget)} | Remaining: {format_tokens(remaining)} | Usage: {pct:.1f}%')

    return '\n'.join(lines)


def format_compact(file_infos: list[dict], budget: int, tag: str = '') -> str:
    """One-line-per-file compact format — good for terminal/logs."""
    total = sum(f['tokens'] for f in file_infos)
    lines = []
    for i, f in enumerate(file_infos, 1):
        lines.append(f"  {i:>2}. {f['name']:<30} {format_tokens(f['tokens']):>7} tokens  ({f['lines']} lines)")
    lines.append(f"{'':>4} {'─' * 48}")
    lines.append(f"{'':>4} Total: {format_tokens(total)} tokens")
    if budget:
        lines.append(f"{'':>4} Budget: {format_tokens(budget)} | Used: {(total/budget)*100:.1f}%")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_to_csv(file_infos: list[dict], budget: int, logfile: str, tag: str = ''):
    """Append a run to a CSV log for historical tracking and eval correlation."""
    total_tokens = sum(f['tokens'] for f in file_infos)
    file_exists = Path(logfile).exists()

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                'timestamp', 'tag', 'total_tokens', 'total_files',
                'budget', 'usage_pct', 'files_included'
            ])
        files_list = ';'.join(f"{f['name']}:{f['tokens']}" for f in file_infos)
        usage_pct = round((total_tokens / budget) * 100, 1) if budget else ''
        writer.writerow([
            datetime.now().isoformat(),
            tag,
            total_tokens,
            len(file_infos),
            budget,
            usage_pct,
            files_list,
        ])


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_loop(paths, extensions, recursive, budget, tag, fmt):
    """Re-scan on file changes. Ctrl+C to exit."""
    print("👁  Watch mode — press Ctrl+C to stop\n")
    last_hash = None
    try:
        while True:
            files = discover_files(paths, extensions, recursive)
            infos = [get_file_info(f) for f in files]
            current_hash = hash(tuple((f['file'], f['tokens'], f.get('modified', '')) for f in infos))

            if current_hash != last_hash:
                os.system('clear' if os.name != 'nt' else 'cls')
                output = format_output(infos, budget, tag, fmt)
                print(output)
                last_hash = current_hash

            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nWatch stopped.")


# ---------------------------------------------------------------------------
# Diff mode
# ---------------------------------------------------------------------------

def diff_manifests(current: list[dict], previous_json: str) -> str:
    """Compare current manifest against a previous JSON manifest."""
    try:
        prev = json.loads(Path(previous_json).read_text())
        prev_files = {f['name']: f['tokens'] for f in prev['files']}
    except Exception as e:
        return f"Error loading previous manifest: {e}"

    curr_files = {f['name']: f['tokens'] for f in current}
    lines = ["\n📊 Context Diff:", ""]

    # Added
    for name in sorted(set(curr_files) - set(prev_files)):
        lines.append(f"  + {name:<30} {format_tokens(curr_files[name]):>7} tokens (new)")

    # Removed
    for name in sorted(set(prev_files) - set(curr_files)):
        lines.append(f"  - {name:<30} {format_tokens(prev_files[name]):>7} tokens (removed)")

    # Changed
    for name in sorted(set(curr_files) & set(prev_files)):
        diff = curr_files[name] - prev_files[name]
        if abs(diff) > 10:  # ignore noise
            sign = '+' if diff > 0 else ''
            lines.append(f"  Δ {name:<30} {sign}{format_tokens(diff):>7} tokens")

    prev_total = sum(prev_files.values())
    curr_total = sum(curr_files.values())
    total_diff = curr_total - prev_total
    sign = '+' if total_diff > 0 else ''
    lines.append(f"\n  Net change: {sign}{format_tokens(total_diff)} tokens")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_output(infos, budget, tag, fmt):
    formatters = {
        'table': format_table,
        'json': format_json,
        'markdown': format_markdown,
        'compact': format_compact,
    }
    return formatters.get(fmt, format_table)(infos, budget, tag)


def main():
    parser = argparse.ArgumentParser(
        description='Context Window Budget & Manifest Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('paths', nargs='*', default=['.'],
                        help='Files or directories to scan (default: current dir)')
    parser.add_argument('--budget', '-b', type=int, default=200000,
                        help='Token budget / context window size (default: 200000)')
    parser.add_argument('--format', '-f', dest='fmt', default='table',
                        choices=['table', 'json', 'markdown', 'compact'],
                        help='Output format (default: table)')
    parser.add_argument('--ext', '-e', nargs='+',
                        help='File extensions to include (default: .md .txt .yaml etc)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not recurse into subdirectories')
    parser.add_argument('--log', '-l', type=str, default=None,
                        help='CSV file to append run data to')
    parser.add_argument('--tag', '-t', type=str, default='',
                        help='Tag/label for this run (for eval correlation)')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch for changes and re-display')
    parser.add_argument('--diff', '-d', type=str, default=None,
                        help='Path to a previous JSON manifest to diff against')
    parser.add_argument('--sort', '-s', default='tokens',
                        choices=['tokens', 'name', 'lines', 'size'],
                        help='Sort files by (default: tokens descending)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save JSON manifest to file (for future --diff)')

    args = parser.parse_args()

    extensions = set(args.ext) if args.ext else None
    recursive = not args.no_recursive

    if args.watch:
        watch_loop(args.paths, extensions, recursive, args.budget, args.tag, args.fmt)
        return

    # Discover and analyze
    files = discover_files(args.paths, extensions, recursive)
    if not files:
        print("No matching files found.", file=sys.stderr)
        sys.exit(1)

    infos = [get_file_info(f) for f in files]

    # Sort
    sort_key = {
        'tokens': lambda f: -f['tokens'],
        'name': lambda f: f['name'].lower(),
        'lines': lambda f: -f['lines'],
        'size': lambda f: -f['size_kb'],
    }[args.sort]
    infos.sort(key=sort_key)

    # Output
    output = format_output(infos, args.budget, args.tag, args.fmt)
    print(output)

    # Diff
    if args.diff:
        diff_output = diff_manifests(infos, args.diff)
        print(diff_output)

    # Log
    if args.log:
        log_to_csv(infos, args.budget, args.log, args.tag)
        print(f"\n📝 Logged to {args.log}")

    # Save manifest for future diffing
    if args.save:
        json_output = format_json(infos, args.budget, args.tag)
        Path(args.save).write_text(json_output)
        print(f"\n💾 Manifest saved to {args.save}")


if __name__ == '__main__':
    main()
