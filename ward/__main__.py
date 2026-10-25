from .cli import main
import sys

# Guarded: multiprocessing's spawn start method re-imports the main module in the worker
# process as __mp_main__; without this guard the CLI would run again inside the worker.
if __name__ == "__main__":
    sys.exit(main())
