"""CCCBDB dataset defaults for the existing shared dipole engine."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parent / 'dipole_152' / 'tools'
sys.path.insert(0, str(ENGINE))
from run_dipole import main as run_main


def main(default_methods=None):
    # Explicit CLI options appear last and override these dataset defaults.
    # Workers use the shared engine's saved metadata, including these geometries.
    original = sys.argv
    sys.argv = [original[0],
                '--input', str(ROOT / 'inputs' / 'cccbdb_inputs.json'),
                '--ref-json', str(ROOT / 'references' / 'ref_cccbdb.json'),
                '--scf-policy', 'standard', *original[1:]]
    try:
        return run_main(default_methods)
    finally:
        sys.argv = original
