#!/usr/bin/env python3
"""UHF/UMP2/OBMP2/OBDH dipoles at CCCBDB experimental geometries."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from cccbdb_driver import main

if __name__ == '__main__':
    sys.exit(main(['uhf', 'ump2', 'obmp2', 'obdh']))
