#!/usr/bin/env python3
"""
Notebook restructuring script for CBU5201_miniproject_2526.ipynb
Implements the approved restructuring plan
"""

import json
import copy
from pathlib import Path

def load_notebook(path):
    """Load Jupyter notebook as JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    """Save Jupyter notebook with proper formatting"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Saved to {path}")

def create_markdown_cell(content):
    """Helper to create a markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def restructure_notebook(input_path, output_path):
    """Main restructuring function"""
    nb = load_notebook(input_path)
    cells = nb['cells']

    print(f"Original notebook: {len(cells)} cells")

    # Track changes
    changes = []

    # ============================================================
    # STEP 1: Structural Fixes
    # ============================================================

    # Step 1.1: Find and delete Cell 4 (孤立的 "## 5. Dataset")
    cells_to_delete = []
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if source.strip() == '## 5. Dataset':
                cells_to_delete.append(i)
                changes.append(f"Delete Cell {i}: Orphaned '## 5. Dataset' title")

    # Delete in reverse order to maintain indices
    for idx in sorted(cells_to_delete, reverse=True):
        del cells[idx]

    # Step 1.2 & 1.3: Fix numbering and unify result summary formats
    # We'll do this by searching for specific patterns

    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])

            # Fix: 5.4 → 5.3 for EDA
            if '## 5.4 EDA' in source or '### 5.4 EDA' in source:
                new_source = source.replace('5.4 EDA', '5.3 EDA')
                cell['source'] = new_source.split('\n')
                changes.append(f"Cell {i}: Renamed 5.4 EDA → 5.3 EDA")

            # Fix: 5.5 → 5.4 for Unsupervised
            if '## 5.5 Unsupervised' in source or '### 5.5 Unsupervised' in source:
                new_source = source.replace('5.5 Unsupervised', '5.4 Unsupervised')
                cell['source'] = new_source.split('\n')
                changes.append(f"Cell {i}: Renamed 5.5 → 5.4 Unsupervised")

            # Unify result summaries: convert bold to headings
            # Look for patterns like "**Observations**:" or "**Key Findings**:"
            if source.strip().startswith('**') and ('Observations' in source or 'Summary' in source or 'Findings' in source or 'Insights' in source):
                # Try to determine the appropriate heading level based on context
                # These are typically sub-sections, so use ###
                if '**Observations' in source and not source.startswith('###'):
                    new_source = source.replace('**Observations', '### Observations').replace('**:', ':')
                    cell['source'] = new_source.split('\n')
                    changes.append(f"Cell {i}: Unified result summary format (bold → heading)")

    # ============================================================
    # STEP 2: Delete Redundant Content
    # ============================================================

    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])

            # Delete 3.2.2 "Comprehensive Course Coverage"
            if '#### 3.2.2' in source and 'Comprehensive Course Coverage' in source:
                # Find the section and delete it
                lines = cell['source']
                new_lines = []
                skip = False
                for line in lines:
                    if '#### 3.2.2' in line and 'Comprehensive' in line:
                        skip = True
                        changes.append(f"Cell {i}: Deleted 3.2.2 'Comprehensive Course Coverage' section")
                    elif line.startswith('####') or line.startswith('###'):
                        skip = False
                    if not skip:
                        new_lines.append(line)
                cell['source'] = new_lines

            # Delete redundant data split table in 5.2 (if exists)
            if '### 5.2' in source or '## 5.2' in source:
                # Replace table with reference
                if '| Split' in source and '| Training' in source:
                    lines = cell['source']
                    new_lines = []
                    in_table = False
                    for line in lines:
                        if '| Split' in line or '| Training' in line or '|---' in line:
                            in_table = True
                        elif in_table and line.strip() and not line.startswith('|'):
                            in_table = False

                        if not in_table or not line.strip().startswith('|'):
                            new_lines.append(line)
                        elif in_table and '| Split' in line:
                            # Replace first line with reference
                            new_lines.append('\n')
                            new_lines.append('*See Section 3.3 for detailed split configuration.*\n')
                            new_lines.append('\n')
                            changes.append(f"Cell {i}: Replaced redundant split table with reference")

                    cell['source'] = new_lines

            # Simplify Cell 1 last paragraph (if contains "After this preliminary analysis")
            if 'After this preliminary analysis' in source:
                # This is a complex text replacement - we'll simplify it
                lines = cell['source']
                new_lines = []
                found = False
                for j, line in enumerate(lines):
                    if 'After this preliminary analysis' in line and not found:
                        # Replace verbose paragraph with concise version
                        new_lines.append('\n')
                        new_lines.append('Based on these findings, I pivoted to a **feature-engineering approach** emphasizing pitch-invariant features and comprehensive model comparison (see Section 3).\n')
                        new_lines.append('\n')
                        found = True
                        changes.append(f"Cell {i}: Simplified verbose paragraph")
                        # Skip lines until next paragraph
                        k = j + 1
                        while k < len(lines) and lines[k].strip() and not lines[k].startswith('#'):
                            k += 1
                        # Continue from there
                        new_lines.extend(lines[k:])
                        break
                    else:
                        new_lines.append(line)

                if found:
                    cell['source'] = new_lines

    # ============================================================
    # STEP 3: Reorder Cells (6.1/6.2 and 6.3/6.4)
    # ============================================================

    # Find indices of sections
    section_indices = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '## 6.1' in source or '### 6.1 Supervised Training' in source:
                section_indices['6.1'] = i
            elif '## 6.2' in source or '### 6.2 Neural' in source:
                section_indices['6.2'] = i
            elif '## 6.3' in source or '### 6.3 Evaluation' in source:
                section_indices['6.3'] = i
            elif '## 6.4' in source or '### 6.4 Model Comparison' in source:
                section_indices['6.4'] = i

    print(f"Found sections: {section_indices}")

    # If 6.2 comes before 6.1, we need to swap them
    if '6.1' in section_indices and '6.2' in section_indices:
        idx_61 = section_indices['6.1']
        idx_62 = section_indices['6.2']

        if idx_62 < idx_61:
            # Find the range of cells for each section
            # 6.2 section: from idx_62 to idx_61-1
            # 6.1 section: from idx_61 to next major section or 6.3

            idx_63 = section_indices.get('6.3', len(cells))

            # Extract sections
            section_62_cells = cells[idx_62:idx_61]
            section_61_cells = cells[idx_61:idx_63]

            # Swap them
            cells[idx_62:idx_63] = section_61_cells + section_62_cells
            changes.append(f"Reordered: Moved 6.1 (cells {idx_61}-{idx_63-1}) before 6.2 (cells {idx_62}-{idx_61-1})")

            # Update indices for 6.3/6.4 swap
            section_indices['6.1'] = idx_62
            section_indices['6.2'] = idx_62 + len(section_61_cells)
            if '6.3' in section_indices:
                section_indices['6.3'] = idx_62 + len(section_61_cells) + len(section_62_cells)

    # Similarly for 6.3/6.4
    if '6.3' in section_indices and '6.4' in section_indices:
        idx_63 = section_indices['6.3']
        idx_64 = section_indices['6.4']

        if idx_64 < idx_63:
            # Find next major section (Chapter 7)
            idx_7 = len(cells)
            for i, cell in enumerate(cells[idx_63:], start=idx_63):
                if cell['cell_type'] == 'markdown':
                    source = ''.join(cell['source'])
                    if '## 7' in source or '# 7' in source:
                        idx_7 = i
                        break

            section_64_cells = cells[idx_64:idx_63]
            section_63_cells = cells[idx_63:idx_7]

            cells[idx_64:idx_7] = section_63_cells + section_64_cells
            changes.append(f"Reordered: Moved 6.3 (cells {idx_63}-{idx_7-1}) before 6.4 (cells {idx_64}-{idx_63-1})")

    # Update notebook
    nb['cells'] = cells

    print(f"\nTotal changes: {len(changes)}")
    for change in changes:
        print(f"  - {change}")

    print(f"\nFinal notebook: {len(cells)} cells")

    return nb

if __name__ == '__main__':
    input_path = Path('/Users/panmingh/Code/ML_Coursework/notebook/CBU5201_miniproject_2526.ipynb')
    output_path = input_path  # Overwrite original (backup already created)

    print("="*60)
    print("Notebook Restructuring Script")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    restructured_nb = restructure_notebook(input_path, output_path)
    save_notebook(restructured_nb, output_path)

    print("\n" + "="*60)
    print("✓ Restructuring complete!")
    print("="*60)
