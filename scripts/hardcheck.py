#!/usr/bin/env python
"""Command-line shim: `python scripts/hardcheck.py runs/<id>` (or `analyst check <id>`).

The checks themselves live in `agentic_analyst.hardcheck` so the eval harness
can import them rather than shelling out to this file and parsing its output.
A script that is also a library is a script whose results nobody has to scrape.
"""

import sys

from agentic_analyst.hardcheck import main

if __name__ == "__main__":
    sys.exit(main())
