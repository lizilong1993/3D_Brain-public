#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--analysis-root',required=True); ap.add_argument('--null-summary',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 ns=json.load(open(a.null_summary)); by={}
 for r in ns: by.setdefault(int(r['seed']),{})[r['null_type']]=r
 rows=[]
 for seed in range(42,52):
  p=Path(a.analysis_root)/f'seed_{seed}/main_B/analysis/results.json'; d=json.load(open(p)); c=d['predicted_regions_nonlesion']['control']['region_id']; rm=json.load(open(p.parent/'regions.json')); nodes=sorted(int(n) for n,v in rm.items() if int(v)==int(c)); iface=set(range(16,24)); groups=by[seed]; vals=[groups[k]['rrp_empirical'] for k in groups]; ps=[groups[k]['p_mc'] for k in groups]
  rows.append({'regime':'formal_main_B_wave1','seed':seed,'candidate_label':'blind_nonlesion_structure_only','candidate_nodes':';'.join(map(str,nodes)),'candidate_size':len(nodes),'iai':len(set(nodes)&iface)/len(nodes),'null_types':';'.join(sorted(groups)),'rrp_min':min(vals),'rrp_by_type':json.dumps({k:groups[k]['rrp_empirical'] for k in groups},sort_keys=True),'p_mc_max':max(ps),'p_mc_by_type':json.dumps({k:groups[k]['p_mc'] for k in groups},sort_keys=True),'null_gate_level2_pass':all(groups[k]['level2_pass'] for k in groups),'full_level2_pass':all(groups[k]['level2_pass'] for k in groups) and len(set(nodes)&iface)==0,'target_rmt_tolerance':1e-8,'candidate_selection_status':'blind_nonlesion_structure_only'})
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print(out)
if __name__=='__main__': main()
