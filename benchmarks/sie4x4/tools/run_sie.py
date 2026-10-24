#!/usr/bin/env python3
"""Compatibility alias; the v1 implementation is run_sie_v1.py."""
import sys
import run_sie_v1 as _v1

if __name__ == "__main__":
    sys.exit(_v1.main())
else:
    sys.modules[__name__] = _v1
