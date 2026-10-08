#!/usr/bin/env python3
"""Run standard SIE WFT and OBDH calculations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_sie_standard import main

if __name__ == '__main__':
    sys.exit(main(['uhf', 'ump2', 'obdh', 'obmp2']))
