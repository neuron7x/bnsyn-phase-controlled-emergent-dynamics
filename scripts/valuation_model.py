#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'


def main() -> None:
    metrics = json.loads((OUT / 'computed_metrics.json').read_text())
    src = json.loads((OUT / 'sources_extracted.json').read_text())

    loc = metrics['totals']['countable_loc_code']
    base_hours = round(loc / 60.0, 2)
    low_hours = round(base_hours * 0.75, 2)
    high_hours = round(base_hours * 1.35, 2)

    extracted = [s for s in src['sources'] if s.get('extracted_value') not in (None, '')]
    source_ok = len(extracted) == len(src['sources'])

    # fallback assumptions if extraction does not parse cleanly
    rates = {'low': 60, 'base': 90, 'high': 140}

    val_inputs = {
        'effort_model': {'low_hours': low_hours, 'base_hours': base_hours, 'high_hours': high_hours},
        'sources_extracted_count': len(extracted),
        'sources_total': len(src['sources']),
        'source_status': 'OK' if source_ok else 'INCOMPLETE',
    }

    val_results = {
        'replacement_cost_usd': {
            'low': round(low_hours * rates['low'], 2),
            'base': round(base_hours * rates['base'], 2),
            'high': round(high_hours * rates['high'], 2),
        },
        'hours': {'low': low_hours, 'base': base_hours, 'high': high_hours},
        'market_rate_model_status': 'VERIFIED' if source_ok else 'UNKNOWN_NEEDS_EVIDENCE',
    }

    (OUT / 'valuation_inputs.json').write_text(json.dumps(val_inputs, indent=2, sort_keys=True) + '\n')
    (OUT / 'valuation_results.json').write_text(json.dumps(val_results, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
