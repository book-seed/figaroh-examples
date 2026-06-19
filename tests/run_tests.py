#!/usr/bin/env python3
"""Test runner — delegates to pytest.

For full validation (tests + example scripts), use:
    python validate.py
"""

import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["tests/", "-v"] + sys.argv[1:]))
