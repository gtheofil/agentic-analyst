"""Back-compatible entry point: `python -m agentic_analyst.run "<brief>"`.

The orchestration moved to `runner.py` and the interface to `cli.py` when the
API arrived and needed the same run logic. This module stays because the README
and the quickstart document it, and a documented command that stops working is
a broken promise to anyone who cloned the repo. It is now three lines of
argument handling over the same two functions `analyst run` calls.
"""

import sys

from agentic_analyst.cli import print_summary, setup_logging
from agentic_analyst.runner import execute_run


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m agentic_analyst.run "<brief>"  (or: analyst run "<brief>")')
        raise SystemExit(1)

    setup_logging()
    record = execute_run(sys.argv[1])
    print_summary(record)
    if record.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
