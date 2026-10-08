#!/usr/bin/env python3
"""Run standard SIE DFT and double-hybrid calculations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_sie_standard import main

if __name__ == '__main__':
    sys.exit(main(['pbe', 'pbe0', 'b3lyp', 'b2plyp', 'dh_matched']))
