#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'/'math_audit'

COEFF={
 'CORE_LOGIC':1.00,
 'TESTS':0.70,
 'CI_CD':0.60,
 'INFRASTRUCTURE':0.50,
 'DOCUMENTATION':0.35,
 'SCRIPTS_TOOLING':0.55,
 'DATA_SCHEMAS':0.60,
 'CONFIGURATION':0.40,
 'STATIC_ASSETS':0.20,
}

def main()->None:
    metrics=json.loads((OUT/'computed_metrics.json').read_text())
    cats=metrics['categories']
    complexity=metrics['complexity']
    comp_factor=1.0
    if complexity.get('radon_average_grade') in ('C','D','E','F'):
        comp_factor=1.25
    test_factor=1.1 if metrics['tests']['test_files']>0 else 1.0

    replacement_hours=0.0
    per_cat={}
    for c,coef in COEFF.items():
        loc=cats[c]['loc_code']
        # base productivity 30 loc/hr normalized by coefficient
        h=(loc/30.0)*coef*comp_factor*test_factor
        per_cat[c]=round(h,2)
        replacement_hours += h

    replacement={
      'low_hours':round(replacement_hours*0.8,2),
      'base_hours':round(replacement_hours,2),
      'high_hours':round(replacement_hours*1.25,2),
    }
    rates_assumption={'low':60,'base':90,'high':140}
    replacement_cost={k:round(replacement[k.replace('rate','hours')]*v,2) for k,v in {'low_rate':60,'base_rate':90,'high_rate':140}.items()}

    valuation_inputs={
      'model_1_repo_replacement':{
        'coefficients':COEFF,
        'complexity_factor':comp_factor,
        'test_factor':test_factor,
        'formula':'hours=sum((loc_code/30)*category_weight*complexity_factor*test_factor)',
        'assumptions':'ASSUMPTION_BOUNDED'
      },
      'model_2_market_rates':{
        'status':'UNKNOWN',
        'reason':'No external rate snapshots collected in this run',
        'required_evidence':['market_rate_source_urls','snapshot_timestamp','fx_source_if_non_usd']
      },
      'external_contributions':{
        'status':'UNKNOWN',
        'required_evidence':['timesheets','invoices','calendar_exports','issue_tracker_links']
      }
    }
    valuation_results={
      'replacement_hours':replacement,
      'replacement_hours_per_category':per_cat,
      'replacement_cost_usd_assumption':{
        'rates_usd_per_hour':rates_assumption,
        'low':round(replacement['low_hours']*rates_assumption['low'],2),
        'base':round(replacement['base_hours']*rates_assumption['base'],2),
        'high':round(replacement['high_hours']*rates_assumption['high'],2),
      },
      'market_rate_model':{'status':'UNKNOWN_NEEDS_EVIDENCE'}
    }
    (OUT/'valuation_inputs.json').write_text(json.dumps(valuation_inputs,indent=2,sort_keys=True)+'\n')
    (OUT/'valuation_results.json').write_text(json.dumps(valuation_results,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
    main()
