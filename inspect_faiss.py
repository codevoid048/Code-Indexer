#!/usr/bin/env python3
"""
FAISS Index Inspector - View FAISS index data in VS Code
Run this to export index data to JSON/CSV files that VS Code can display
"""

from datetime import datetime
import json
import csv
from pathlib import Path

from core.vectorstore import TextBasedVectorStore


def inspect_faiss_indices():
    """Inspect and export FAISS index data to viewable formats."""

    print("Inspecting FAISS Indices...")
    print("=" * 50)

    # Initialize vector store (loads existing indices)
    store = TextBasedVectorStore()

    # Create output directory
    output_dir = Path("index_inspection")
    output_dir.mkdir(exist_ok=True)

    # Export symbol data
    if store.symbol_metadata:
        print(f"Exporting {len(store.symbol_metadata)} symbols...")

        # JSON export
        symbol_data = []
        for symbol_id, symbol in store.symbol_metadata.items():
            symbol_dict = symbol.model_dump()
            # Convert datetime to string for JSON
            if 'last_modified' in symbol_dict and symbol_dict['last_modified']:
                symbol_dict['last_modified'] = symbol_dict['last_modified'].isoformat()
            if 'last_indexed' in symbol_dict and symbol_dict['last_indexed']:
                symbol_dict['last_indexed'] = symbol_dict['last_indexed'].isoformat()

            symbol_data.append(symbol_dict)

        with open(output_dir / "symbols.json", 'w', encoding='utf-8') as f:
            json.dump(symbol_data, f, indent=2, ensure_ascii=False)

        # CSV export
        if symbol_data:
            fieldnames = symbol_data[0].keys()
            with open(output_dir / "symbols.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(symbol_data)

        print(f"[✓]Symbols exported to: {output_dir}/symbols.json & symbols.csv")

    # Export file data
    if store.file_metadata:
        print(f"Exporting {len(store.file_metadata)} files...")

        file_data = []
        for file_id, code_file in store.file_metadata.items():
            file_dict = code_file.model_dump()
            # Convert datetime to string for JSON
            if 'last_modified' in file_dict and file_dict['last_modified']:
                file_dict['last_modified'] = file_dict['last_modified'].isoformat()
            if 'last_indexed' in file_dict and file_dict['last_indexed']:
                file_dict['last_indexed'] = file_dict['last_indexed'].isoformat()

            file_data.append(file_dict)

        with open(output_dir / "files.json", 'w', encoding='utf-8') as f:
            json.dump(file_data, f, indent=2, ensure_ascii=False)

        # CSV export
        if file_data:
            fieldnames = file_data[0].keys()
            with open(output_dir / "files.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(file_data)

        print(f"[✓]Files exported to: {output_dir}/files.json & files.csv")

    # Export vector mappings
    if store.symbol_strings:
        print(f"Exporting {len(store.symbol_strings)} symbol string mappings...")

        with open(output_dir / "symbol_mappings.json", 'w', encoding='utf-8') as f:
            json.dump(store.symbol_strings, f, indent=2, ensure_ascii=False)

        print(f"[✓]Symbol mappings exported to: {output_dir}/symbol_mappings.json")

    if store.file_strings:
        print(f"Exporting {len(store.file_strings)} file string mappings...")

        with open(output_dir / "file_mappings.json", 'w', encoding='utf-8') as f:
            json.dump(store.file_strings, f, indent=2, ensure_ascii=False)

        print(f"[✓]File mappings exported to: {output_dir}/file_mappings.json")

    # Export string to ID mappings
    if store.string_to_id:
        print(f"Exporting {len(store.string_to_id)} string-to-ID mappings...")

        with open(output_dir / "string_to_id.json", 'w', encoding='utf-8') as f:
            json.dump(store.string_to_id, f, indent=2, ensure_ascii=False)

        print(f"[✓]String-to-ID mappings exported to: {output_dir}/string_to_id.json")

    # Create a summary
    summary = {
        "total_symbols": len(store.symbol_metadata),
        "total_files": len(store.file_metadata),
        "symbol_index_size": len(store.symbol_strings),
        "file_index_size": len(store.file_strings),
        "export_timestamp": datetime.now(),
        "exported_files": [
            "symbols.json", "symbols.csv",
            "files.json", "files.csv",
            "symbol_mappings.json", "file_mappings.json",
            "string_to_id.json"
        ]
    }

    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary exported to: {output_dir}/summary.json")
    print()
    print("Export complete! Open these files in VS Code:")
    print(f"   {output_dir}/")
    print("   ├── symbols.json (all symbol data)")
    print("   ├── symbols.csv (spreadsheet view)")
    print("   ├── files.json (all file data)")
    print("   ├── files.csv (spreadsheet view)")
    print("   ├── symbol_mappings.json (text → vector mappings)")
    print("   ├── file_mappings.json (text → vector mappings)")
    print("   ├── string_to_id.json (lookup table)")
    print("   └── summary.json (overview)")


if __name__ == "__main__":
    try:
        inspect_faiss_indices()
    except Exception as e:
        print(f"[X] Error inspecting indices: {e}")
        import traceback
        traceback.print_exc()