import os
import re
from collections import defaultdict


def parse_cif_structure(cif_file_path: str):
    """
    Parse CIF file and extract all categories and their fields.
    
    Args:
        cif_file_path: Path to the CIF file
        
    Returns:
        dict: Dictionary of categories and their fields with sample values
    """
    categories = defaultdict(dict)
    current_loop_fields = []
    in_loop = False
    
    with open(cif_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('loop_'):
                in_loop = True
                current_loop_fields = []
                continue
            
            if line.startswith('_'):
                parts = line.split(None, 1)
                field = parts[0]
                value = parts[1] if len(parts) > 1 else None
                
                category = field.rsplit('.', 1)[0] if '.' in field else field
                field_name = field.rsplit('.', 1)[1] if '.' in field else field
                
                if in_loop:
                    current_loop_fields.append(field)
                    categories[category][field_name] = {
                        'type': 'loop',
                        'sample': None
                    }
                else:
                    categories[category][field_name] = {
                        'type': 'single',
                        'sample': value
                    }
            elif in_loop and current_loop_fields:
                in_loop = False
                current_loop_fields = []
    
    return dict(categories)


def display_cif_structure(cif_file_path: str, show_samples: bool = True, max_categories: int = None):
    """
    Display the structure of a CIF file in a readable format.
    
    Args:
        cif_file_path: Path to the CIF file
        show_samples: Whether to show sample values
        max_categories: Maximum number of categories to display (None for all)
    """
    print(f"\n{'='*80}")
    print(f"CIF File: {os.path.basename(cif_file_path)}")
    print(f"{'='*80}\n")
    
    categories = parse_cif_structure(cif_file_path)
    
    print(f"Total categories found: {len(categories)}\n")
    
    category_list = list(categories.items())
    if max_categories:
        category_list = category_list[:max_categories]
    
    for cat_name, fields in category_list:
        print(f"\n{cat_name}")
        print(f"{'-'*len(cat_name)}")
        print(f"Fields: {len(fields)}")
        
        if show_samples and len(fields) <= 20:
            for field_name, info in sorted(fields.items())[:10]:
                field_type = info['type']
                sample = info['sample']
                
                if sample and len(str(sample)) > 50:
                    sample = str(sample)[:47] + '...'
                
                print(f"  • {field_name} [{field_type}]", end='')
                if sample and field_type == 'single':
                    print(f": {sample}")
                else:
                    print()
            
            if len(fields) > 10:
                print(f"  ... and {len(fields) - 10} more fields")
        else:
            field_names = sorted(list(fields.keys()))[:5]
            print(f"  Fields: {', '.join(field_names)}", end='')
            if len(fields) > 5:
                print(f", ... ({len(fields) - 5} more)")
            else:
                print()
    
    if max_categories and len(categories) > max_categories:
        print(f"\n... and {len(categories) - max_categories} more categories")
    
    print(f"\n{'='*80}\n")


def list_all_categories(cif_file_path: str):
    """
    List all category names in a CIF file.
    
    Args:
        cif_file_path: Path to the CIF file
        
    Returns:
        list: Sorted list of all category names
    """
    categories = parse_cif_structure(cif_file_path)
    return sorted(categories.keys())


def get_category_fields(cif_file_path: str, category_name: str):
    """
    Get all fields for a specific category.
    
    Args:
        cif_file_path: Path to the CIF file
        category_name: Name of the category (e.g., '_atom_site')
        
    Returns:
        dict: Dictionary of fields and their info
    """
    categories = parse_cif_structure(cif_file_path)
    return categories.get(category_name, {})


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inspect_cif.py <cif_file_path> [--list-categories] [--category <name>]")
        print("\nExamples:")
        print("  python inspect_cif.py file.cif")
        print("  python inspect_cif.py file.cif --list-categories")
        print("  python inspect_cif.py file.cif --category _atom_site")
        sys.exit(1)
    
    cif_file = sys.argv[1]
    
    if not os.path.exists(cif_file):
        print(f"Error: File not found: {cif_file}")
        sys.exit(1)
    
    if '--list-categories' in sys.argv:
        categories = list_all_categories(cif_file)
        print(f"\nAll categories in {os.path.basename(cif_file)}:")
        print(f"{'='*80}")
        for i, cat in enumerate(categories, 1):
            print(f"{i:3d}. {cat}")
        print(f"\nTotal: {len(categories)} categories\n")
    
    elif '--category' in sys.argv:
        idx = sys.argv.index('--category')
        if idx + 1 < len(sys.argv):
            category = sys.argv[idx + 1]
            fields = get_category_fields(cif_file, category)
            
            if fields:
                print(f"\nCategory: {category}")
                print(f"{'='*80}")
                print(f"Total fields: {len(fields)}\n")
                
                for field_name, info in sorted(fields.items()):
                    print(f"  • {field_name} [{info['type']}]", end='')
                    if info['sample']:
                        sample = str(info['sample'])
                        if len(sample) > 60:
                            sample = sample[:57] + '...'
                        print(f": {sample}")
                    else:
                        print()
            else:
                print(f"Category '{category}' not found")
        else:
            print("Error: --category requires a category name")
    
    else:
        display_cif_structure(cif_file, show_samples=True, max_categories=20)
