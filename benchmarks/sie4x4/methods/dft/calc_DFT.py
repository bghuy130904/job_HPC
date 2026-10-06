#!/usr/bin/env python3
"""Compatibility entry point; shared SIE protocol lives in tools/run_sie.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_sie import main
if __name__ == '__main__':
    sys.exit(main(['pbe', 'pbe0', 'dh_matched']))
