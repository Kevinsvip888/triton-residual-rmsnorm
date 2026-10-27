#!/usr/bin/env python3
"""Restore only the published policy for a demo, refusing different local state."""
from pathlib import Path
from verify_public_snapshot import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    verify(ROOT, verbose=False)
    source = ROOT / 'results/warp_tuning/selected_policy.json'
    destination = ROOT / 'outputs/day11/warp_tuning_v1/selected_policy.json'
    value = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != value:
            raise RuntimeError('A different local selected policy exists; keep it unchanged and use a separate checkout for the published demo')
        print('PASS: existing selected policy matches the published bytes; no change.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacement if another process creates it first.
    with destination.open('xb') as handle:
        handle.write(value)
    print('PASS: published selected policy restored for the demo.')
    print('No validation receipts, benchmark protocols, or completion flags were created.')


if __name__ == '__main__':
    main()
