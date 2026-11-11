#!/usr/bin/env python3
"""CLI-обёртка над ScannerCore (без GUI)."""
import time
from scanner_core import ScannerCore

def main():
    scanner = ScannerCore(
        x_zone=52, y_zone=30,
        x_step=1, y_step=1,
        t_var=0.01, cycles=1, both_sides=False,
        dev=None,
    )
    scanner.on_status = lambda s: print(f"[status] {s}")
    scanner.on_step = lambda p, t, w: print(f"[step] {p}/{t}" + (" (waiting)" if w else ""))
    scanner.on_finished = lambda: print("[finished]")
    scanner.on_error = lambda e: print(f"[error] {e}")

    scanner.start()
    while scanner.running:
        scanner.poll()
        time.sleep(max(0.01, scanner.t_var))

if __name__ == "__main__":
    main()
