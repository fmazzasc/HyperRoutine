#!/usr/bin/env python3
"""Chi2-only entry point; accepts the same JSON configuration as spectrum.py.

The reusable computation is utils.compute_model_chi2. Legacy individual CLI
options are replaced by spectrum_config.json settings.
"""
import sys

from spectrum import main
from utils import Chi2Result, compute_model_chi2, print_result


if __name__ == '__main__':
    main([*sys.argv[1:], '--chi2-only'])
