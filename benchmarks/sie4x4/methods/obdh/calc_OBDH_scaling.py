#!/usr/bin/env python3
"""Entry point for the explicitly versioned SIE scaling_v2 protocol."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_sie_scaling import main

if __name__ == '__main__':
    sys.exit(main())
