"""Combine completed run tables without replacing missing energies with zero."""
import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, columns = [], ['Run_dir', 'Method']
    for directory in args.run_dirs:
        with (directory / 'interaction_energies.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                # Preserve Status and all existing blank energy cells.
                row = {'Run_dir': str(directory), 'Method': directory.parent.name, **row}
                for key in row:
                    if key not in columns:
                        columns.append(key)
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    main()
