#!/usr/bin/env python3
"""
Parquet Dataset Reader Utility

Usage: python preader.py <filepath> [--row N]

Displays dataset information including:
- Dataset shape and size
- Column names and types
- Specified row data with formatted output
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


class ValueInfo:
    """Cache for value analysis to avoid redundant type checking."""
    __slots__ = ('value', 'type_name', 'shape_info', 'is_dict', 'is_ndarray', 
                 'is_list', 'is_str', 'is_primitive', 'length')
    
    def __init__(self, value):
        self.value = value
        self.is_dict = isinstance(value, dict)
        self.is_ndarray = isinstance(value, np.ndarray)
        self.is_str = isinstance(value, str)
        self.is_primitive = isinstance(value, (int, float, bool, np.integer, np.floating))
        self.is_list = hasattr(value, '__len__') and not (self.is_str or self.is_dict)
        
        # Calculate once
        if self.is_dict:
            self.length = len(value)
            self.type_name = "dict"
            self.shape_info = f"dict (keys={self.length})"
        elif self.is_ndarray:
            self.length = value.size
            self.type_name = f"ndarray"
            self.shape_info = f"ndarray {value.shape} [{value.dtype}]"
        elif self.is_str:
            self.length = len(value)
            self.type_name = "str"
            self.shape_info = f"str len={self.length}"
        elif self.is_list:
            try:
                self.length = len(value)
                if self.length > 0 and hasattr(value[0], '__len__') and not isinstance(value[0], str):
                    if hasattr(value[0], 'shape'):
                        self.shape_info = f"list[array] len={self.length}, first={value[0].shape}"
                    else:
                        self.shape_info = f"list[list] len={self.length}"
                else:
                    self.shape_info = f"list len={self.length}"
                self.type_name = "list"
            except (TypeError, IndexError, KeyError):
                self.shape_info = type(value).__name__
                self.type_name = type(value).__name__
                self.length = 0
        elif self.is_primitive:
            self.length = 0
            self.type_name = type(value).__name__
            self.shape_info = self.type_name
        elif value is None:
            self.length = 0
            self.type_name = "None"
            self.shape_info = "None"
        else:
            self.length = 0
            self.type_name = type(value).__name__
            self.shape_info = self.type_name


def print_keys_tree(d, indent=0, prefix=""):
    """
    Print a recursive tree of keys without showing content.
    
    Args:
        d: Dictionary to print
        indent: Current indentation level
        prefix: Prefix for the current key
    """
    indent_str = "  " * indent
    items = list(d.items())  # Convert once
    num_items = len(items)
    
    for i, (key, val) in enumerate(items):
        is_last = (i == num_items - 1)
        branch = "└─ " if is_last else "├─ "
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(val, dict):
            print(f"{indent_str}{branch}{key}: dict")
            print_keys_tree(val, indent, prefix=full_key)
        elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], dict):
            print(f"{indent_str}{branch}{key}: list[dict] (length={len(val)})")
        elif isinstance(val, (list, tuple)):
            print(f"{indent_str}{branch}{key}: list (length={len(val)})")
        elif isinstance(val, np.ndarray):
            print(f"{indent_str}{branch}{key}: ndarray(shape={val.shape}, dtype={val.dtype})")
        elif isinstance(val, str):
            print(f"{indent_str}{branch}{key}: str (length={len(val)})")
        else:
            print(f"{indent_str}{branch}{key}: {type(val).__name__}")


def print_shape_table(value_infos):
    """
    Print a compact table showing shape/count information for each key.
    Uses pre-computed ValueInfo objects to avoid re-scanning.
    
    Args:
        value_infos: List of tuples (key, ValueInfo)
    """
    if not value_infos:
        return
    
    # Calculate column width once
    max_key_len = max((len(str(key)) for key, _ in value_infos), default=10)
    max_key_len = max(max_key_len, 10)
    
    # Print table - data already computed in ValueInfo
    print(f"{'Key':<{max_key_len}} | Shape/Type")
    print(f"{'-' * max_key_len}-+-{'-' * 40}")
    for key, info in value_infos:
        print(f"{key:<{max_key_len}} | {info.shape_info}")


def truncate_string(s, max_len=500, middle_keep=100):
    """
    Truncate a string by omitting the middle part if too long.
    
    Args:
        s: String to truncate
        max_len: Maximum length before truncation
        middle_keep: How many chars to keep from start and end
        
    Returns:
        str: Truncated string or original if short enough
    """
    if len(s) <= max_len:
        return s
    
    keep_each = middle_keep // 2
    omitted = len(s) - middle_keep
    return f"{s[:keep_each]}\n... ({omitted} chars omitted) ...\n{s[-keep_each:]}"


def format_dict_recursively(d, indent=0, max_depth=10, key_name=""):
    """
    Format a dictionary recursively with proper indentation and smart truncation.
    
    Args:
        d: Dictionary to format
        indent: Current indentation level
        max_depth: Maximum recursion depth
        key_name: Name of the current key (for special handling)
        
    Returns:
        str: Formatted dictionary string
    """
    if indent > max_depth:
        return "... (max depth reached)"
    
    if not isinstance(d, dict):
        return str(d)
    
    lines = []
    indent_str = "  " * indent
    
    # Process items once
    items = list(d.items())
    for key, val in items:
        if isinstance(val, dict):
            nested = format_dict_recursively(val, indent + 1, max_depth, key)
            lines.append(f"{indent_str}{key}: {{\n{nested}\n{indent_str}}}")
        elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], dict):
            lines.append(f"{indent_str}{key}: [")
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    nested = format_dict_recursively(item, indent + 2, max_depth, key)
                    lines.append(f"{indent_str}  [{i}]: {{\n{nested}\n{indent_str}  }}")
                else:
                    lines.append(f"{indent_str}  [{i}]: {item}")
            lines.append(f"{indent_str}]")
        elif isinstance(val, str):
            # Special handling for 'messages' - show more content
            if key == 'messages' or 'message' in key.lower():
                truncated = truncate_string(val, max_len=2000, middle_keep=400)
            else:
                truncated = truncate_string(val, max_len=500, middle_keep=100)
            lines.append(f"{indent_str}{key}: '{truncated}'")
        elif isinstance(val, (list, tuple)):
            lines.append(f"{indent_str}{key}: {val}")
        else:
            lines.append(f"{indent_str}{key}: {val}")
    
    return "\n".join(lines)


def format_value_with_info(info, key_name=""):
    """
    Format a value for display using pre-computed ValueInfo.
    
    Args:
        info: ValueInfo object with cached type analysis
        key_name: Name of the key (for special handling)
        
    Returns:
        str: Formatted string representation
    """
    value = info.value
    
    # Handle dictionaries recursively
    if info.is_dict:
        formatted = format_dict_recursively(value, key_name=key_name)
        return f"{{\n{formatted}\n}}"
    
    # Handle numpy arrays - show like numpy output
    elif info.is_ndarray:
        if info.length == 0:
            return f"array([], shape={value.shape}, dtype={value.dtype})"
        elif info.length <= 20:
            return f"array({str(value)}, dtype={value.dtype})"
        else:
            # Show numpy-style abbreviated output
            with np.printoptions(threshold=10, edgeitems=3):
                return f"array({str(value)}, dtype={value.dtype})"
    
    # Handle pandas arrays or object dtype containing arrays
    elif info.is_list:
        try:
            if info.length > 0 and hasattr(value[0], '__len__') and not isinstance(value[0], str):
                if hasattr(value[0], 'shape'):
                    # Array of numpy arrays
                    return f"[array(shape={value[0].shape}), ...] (total: {info.length} arrays)"
                else:
                    return f"[[...], ...] (total: {info.length} lists)"
            elif info.length <= 20:
                return str(value)
            else:
                return f"[{value[0]}, {value[1]}, ..., {value[-1]}] (length={info.length})"
        except (TypeError, IndexError, KeyError):
            return str(value)
    
    # Handle strings - show with smart truncation
    elif info.is_str:
        # Special handling for 'messages' key
        if key_name == 'messages' or 'message' in key_name.lower():
            truncated = truncate_string(value, max_len=2000, middle_keep=400)
        else:
            truncated = truncate_string(value, max_len=500, middle_keep=100)
        return f"'{truncated}'"
    
    # Handle numbers and other primitives
    elif info.is_primitive:
        return str(value)
    
    # Handle None
    elif value is None:
        return "None"
    
    # Default case
    else:
        return str(value)


def display_row(df, row_index, filepath_name, is_first_view=True):
    """
    Display a specific row from the dataframe.
    
    Args:
        df: DataFrame to display from
        row_index: Index of the row to display
        filepath_name: Name of the file for header
        is_first_view: Whether this is the first view (show tree) or subsequent (show table)
    """
    if row_index < 0 or row_index >= len(df):
        print(f"\nError: Row index {row_index} is out of range. Dataset has {len(df)} rows (0-{len(df)-1}).\n")
        return False
    
    selected_row = df.iloc[row_index]
    
    # Single pass: analyze all values once and cache the info
    value_infos = [(col, ValueInfo(selected_row[col])) for col in df.columns]
    
    print("\n" + "=" * 80)
    print(f"DATA ENTRY (Row {row_index} of {len(df)-1})")
    print("=" * 80)
    
    # Display formatted values using cached info
    for col, info in value_infos:
        formatted_val = format_value_with_info(info, key_name=col)
        print(f"\n[{col}]")
        print(formatted_val)
    
    print()
    
    # Essential info
    print("=" * 80)
    print(f"PARQUET DATASET: {filepath_name}")
    print("=" * 80)
    print(f"Data count: {len(df)} rows (0-{len(df)-1})")
    print()
    
    # Show structure: tree on first view, compact table on subsequent views
    print("=" * 80)
    if is_first_view:
        print("DATA STRUCTURE (Keys Tree)")
        print("=" * 80)
        # For tree view, need the actual dict
        row_dict = selected_row.to_dict()
        print_keys_tree(row_dict)
    else:
        print("DATA STRUCTURE (Shape/Count Table)")
        print("=" * 80)
        # Use cached info instead of re-analyzing
        print_shape_table(value_infos)
    
    print()
    print("=" * 80)
    
    return True


def analyze_parquet_interactive(filepath, initial_row=0):
    """
    Analyze a parquet file interactively, allowing user to browse rows.
    
    Args:
        filepath: Path to the parquet file
        initial_row: Initial row index to display
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    if not filepath.suffix == '.parquet':
        print(f"Warning: File does not have .parquet extension: {filepath}")
    
    # Read the parquet file
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print(f"PARQUET READER - {filepath.name}")
    print("=" * 80)
    print(f"File size: {filepath.stat().st_size / (1024**2):.2f} MB")
    print(f"Total rows: {len(df)} (indices: 0-{len(df)-1})")
    print("=" * 80)
    
    # Display initial row with structure tree
    current_row = initial_row
    if not display_row(df, current_row, filepath.name, is_first_view=True):
        sys.exit(1)
    
    # Interactive loop
    while True:
        print("\nOptions:")
        print(f"  - Enter row number (0-{len(df)-1}) to view that row")
        print("  - Enter 'n' or 'next' to view next row")
        print("  - Enter 'p' or 'prev' to view previous row")
        print("  - Enter 'q', 'quit', or 'exit' to quit")
        
        try:
            user_input = input("\n> ").strip().lower()
            
            if user_input in ['q', 'quit', 'exit', '']:
                print("\nExiting. Goodbye!")
                break
            
            elif user_input in ['n', 'next']:
                current_row = min(current_row + 1, len(df) - 1)
                display_row(df, current_row, filepath.name, is_first_view=False)
            
            elif user_input in ['p', 'prev', 'previous']:
                current_row = max(current_row - 1, 0)
                display_row(df, current_row, filepath.name, is_first_view=False)
            
            else:
                # Try to parse as row number
                try:
                    new_row = int(user_input)
                    if display_row(df, new_row, filepath.name, is_first_view=False):
                        current_row = new_row
                except ValueError:
                    print(f"\nInvalid input: '{user_input}'. Please enter a number or command.\n")
        
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting. Goodbye!")
            break


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Parquet Dataset Reader - Inspect parquet files with detailed formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preader.py data/file.parquet              # Start interactive viewer at row 0
  python preader.py data/file.parquet --row 5      # Start interactive viewer at row 5
  python preader.py data/file.parquet -r 10        # Start interactive viewer at row 10

Interactive Commands:
  - Enter a number to view that row
  - 'n' or 'next' - view next row
  - 'p' or 'prev' - view previous row  
  - 'q' or 'quit' - exit the viewer
        """
    )
    
    parser.add_argument('filepath', help='Path to the parquet file')
    parser.add_argument('-r', '--row', type=int, default=0,
                       help='Starting row index to display (0-based, default: 0)')
    
    args = parser.parse_args()
    
    analyze_parquet_interactive(args.filepath, args.row)


if __name__ == "__main__":
    main()
