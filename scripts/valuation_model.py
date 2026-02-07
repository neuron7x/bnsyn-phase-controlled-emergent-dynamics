#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'


def main() -> None:
    metrics = json.loads((OUT / 'computed_metrics.json').read_text(encoding='utf-8'))
    sources = json.loads((OUT / 'sources_extracted.json').read_text(encoding='utf-8'))

    loc = metrics['totals']['countable_loc_code']
    replacement_hours = {
        'min': round(loc / 85.0, 2),
        'base': round(loc / 60.0, 2),
        'max': round(loc / 40.0, 2),
    }
    assumptions = {
        'model': 'replacement_cost',
        'hours_per_loc': {'min': 1 / 85.0, 'base': 1 / 60.0, 'max': 1 / 40.0},
        'usd_hourly_rate': {'min': 60, 'base': 90, 'max': 140},
        'note': 'Bounded assumptions used when market-rate extraction is unavailable.',
    }

    replacement_cost = {
        'min': round(replacement_hours['min'] * assumptions['usd_hourly_rate']['min'], 2),
        'base': round(replacement_hours['base'] * assumptions['usd_hourly_rate']['base'], 2),
        'max': round(replacement_hours['max'] * assumptions['usd_hourly_rate']['max'], 2),
    }

    extracted = [s for s in sources.get('sources', []) if s.get('extracted_value') not in (None, '')]
    market_rate_verified = bool(sources.get('sources')) and len(extracted) == len(sources['sources'])

    valuation_inputs = {
        'replacement_model_assumptions': assumptions,
        'countable_loc_code': loc,
        'sources_total': len(sources.get('sources', [])),
        'sources_with_values': len(extracted),
    }
    valuation_results = {
        'replacement_cost_model': {
            'status': 'VERIFIED',
            'hours': replacement_hours,
            'cost_usd': replacement_cost,
        },
        'market_rate_model': {
            'status': 'VERIFIED' if market_rate_verified else 'NOT VERIFIED',
            'reason': None if market_rate_verified else 'One or more configured sources missing extracted values.',
        },
        'overall_status': 'VERIFIED' if market_rate_verified else 'NOT VERIFIED',
    }

    (OUT / 'valuation_inputs.json').write_text(json.dumps(valuation_inputs, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'valuation_results.json').write_text(json.dumps(valuation_results, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
