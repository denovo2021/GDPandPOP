#!/usr/bin/env python
"""
update_imports.py - Update import statements after project reorganization

This script updates all Python files to use the new src-based import structure
and updates config.py to reflect the new directory layout.

Usage:
    python update_imports.py --dry-run    # Preview changes
    python update_imports.py              # Execute updates
"""

import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# Import replacement patterns
# Format: (old_pattern, new_replacement)
IMPORT_REPLACEMENTS = [
    # Config imports
    (r'from config import', 'from src.config import'),
    (r'import config\b', 'import src.config as config'),

    # Models imports
    (r'from models\.', 'from src.models.'),
    (r'from models import', 'from src.models import'),

    # Analysis imports
    (r'from analysis\.', 'from src.analysis.'),
    (r'from analysis import', 'from src.analysis import'),

    # Visualization imports
    (r'from visualization\.', 'from src.visualization.'),
    (r'from visualization import', 'from src.visualization import'),

    # Data processing imports
    (r'from data_processing\.', 'from src.data_processing.'),
    (r'from data_processing import', 'from src.data_processing import'),

    # Utils imports
    (r'from utils\.', 'from src.utils.'),
    (r'from utils import', 'from src.utils import'),

    # Tables imports
    (r'from tables\.', 'from src.tables.'),
    (r'from tables import', 'from src.tables import'),

    # Scripts imports
    (r'from scripts\.', 'from src.scripts.'),
    (r'from scripts import', 'from src.scripts import'),
]

# Path replacements in config.py
CONFIG_PATH_REPLACEMENTS = [
    # Data directory: MacroMetrics -> data
    (r'DIR_DATA = PROJECT_ROOT / "MacroMetrics"', 'DIR_DATA = PROJECT_ROOT / "data"'),

    # Output directory: outputs -> results
    (r'DIR_OUTPUT = PROJECT_ROOT / "outputs"', 'DIR_OUTPUT = PROJECT_ROOT / "results"'),
]


def update_file_imports(filepath: Path, dry_run: bool = False) -> tuple[bool, list]:
    """Update imports in a single Python file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            content = filepath.read_text(encoding='cp932')
        except:
            return False, []

    original_content = content
    changes = []

    for pattern, replacement in IMPORT_REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            # Find what changed
            changes.append(f"  {pattern} -> {replacement}")
            content = new_content

    if content != original_content:
        if not dry_run:
            filepath.write_text(content, encoding='utf-8')
        return True, changes

    return False, []


def update_config_paths(dry_run: bool = False) -> bool:
    """Update path definitions in config.py."""
    config_path = PROJECT_ROOT / "src" / "config.py"

    if not config_path.exists():
        # Try old location
        config_path = PROJECT_ROOT / "config.py"

    if not config_path.exists():
        print("  [WARN] config.py not found")
        return False

    content = config_path.read_text(encoding='utf-8')
    original_content = content

    for pattern, replacement in CONFIG_PATH_REPLACEMENTS:
        content = content.replace(pattern, replacement)

    if content != original_content:
        if not dry_run:
            config_path.write_text(content, encoding='utf-8')
        return True

    return False


def update_sys_path_inserts(filepath: Path, dry_run: bool = False) -> tuple[bool, list]:
    """Update sys.path.insert statements to work with new structure."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except:
        return False, []

    original_content = content
    changes = []

    # Pattern for sys.path.insert that adds parent directory
    # These need to be updated to add the project root (parent.parent for src/subdir/)
    old_patterns = [
        (r'sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\.parent\)\)',
         'sys.path.insert(0, str(Path(__file__).parent.parent.parent))'),
        (r'sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\)\)',
         'sys.path.insert(0, str(Path(__file__).parent.parent))'),
    ]

    for pattern, replacement in old_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f"  Updated sys.path.insert")

    if content != original_content:
        if not dry_run:
            filepath.write_text(content, encoding='utf-8')
        return True, changes

    return False, []


def main():
    parser = argparse.ArgumentParser(description="Update imports after reorganization")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    args = parser.parse_args()

    print("=" * 60)
    print("Updating Python imports for new project structure")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Find all Python files in src/
    src_dir = PROJECT_ROOT / "src"
    if not src_dir.exists():
        print("[ERROR] src/ directory not found. Run organize_project.py first.")
        return

    python_files = list(src_dir.rglob("*.py"))

    # Also check root level files
    for f in PROJECT_ROOT.glob("*.py"):
        if f.name not in ["organize_project.py", "update_imports.py"]:
            python_files.append(f)

    print(f"\n[1/3] Updating imports in {len(python_files)} Python files...")

    updated_count = 0
    for filepath in python_files:
        relative_path = filepath.relative_to(PROJECT_ROOT)
        changed, changes = update_file_imports(filepath, args.dry_run)

        if changed:
            updated_count += 1
            print(f"  [UPDATE] {relative_path}")
            for change in changes[:3]:  # Show first 3 changes
                print(f"    {change}")
            if len(changes) > 3:
                print(f"    ... and {len(changes) - 3} more")

    print(f"\n  Updated {updated_count} files")

    print("\n[2/3] Updating config.py paths...")
    if update_config_paths(args.dry_run):
        print("  [UPDATE] config.py - Updated data/output paths")
    else:
        print("  [SKIP] config.py - No changes needed or file not found")

    print("\n[3/3] Updating sys.path.insert statements...")
    syspath_count = 0
    for filepath in python_files:
        relative_path = filepath.relative_to(PROJECT_ROOT)
        changed, changes = update_sys_path_inserts(filepath, args.dry_run)
        if changed:
            syspath_count += 1
            print(f"  [UPDATE] {relative_path}")

    print(f"\n  Updated {syspath_count} files")

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE - Run without --dry-run to execute")
    else:
        print("IMPORT UPDATES COMPLETE")
        print("\nRemaining manual steps:")
        print("  1. Review changes in key files")
        print("  2. Run tests to verify imports work")
        print("  3. Update any hardcoded paths in scripts")
    print("=" * 60)


if __name__ == "__main__":
    main()
