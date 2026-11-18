"""Cold-path evidence parser shared by standalone candidate launchers."""
import re

ROW = re.compile(r'^\s*(\d+)\s+(free|busy|used|acquired)\s+(\d+)\s+(\S+)\s+(0x[0-9a-fA-F]+)\s+(\S+)',re.M)

def parse_snapshot(text):
    records=[]
    for match in ROW.finditer(text):
        block,state,size,physical,mask,owner=match.groups()
        records.append({'id':int(block),'state':state,'size':int(size),
            'physical':physical,'cpu_mask':int(mask,16),
            'owner':int(owner) if owner.isdigit() else None})
    return {'real_v2_available':all(token in text for token in ('backend=v2','runtime=available','fake=no')),
            'blocks':records}

def validate(samples,rows):
    all_owners={int(owner) for row in rows for owner in row.get('worker_tids',[])}
    all_owners.update(int(row['process_pid']) for row in rows if 'process_pid' in row)
    required_sets={tuple(sorted(row.get('tcm_block_ids',[]))) for row in rows}
    six_claims=bool(rows) and all(len(ids)==6 and len(set(ids))==6 and min(ids)>=0 for ids in required_sets)
    ownership=bool(rows) and all(row.get('tcm_ownership_errors')==0 for row in rows) and six_claims
    observed=[]
    for number,sample in enumerate(samples):
        parsed=parse_snapshot(sample.get('stdout',''))
        if sample.get('returncode')!=0 or not parsed['real_v2_available']:continue
        busy={block['id']:block for block in parsed['blocks'] if block['state']!='free'}
        for ids in required_sets:
            if len(ids)!=6 or len(set(ids))!=6 or not all(block in busy for block in ids):continue
            if not all(busy[block]['owner'] in all_owners for block in ids):continue
            # The six observed blocks must be usable on the actual AI6 set;
            # the worker's production lease additionally checks its exact CPU.
            if not all(busy[block]['cpu_mask'] & sum(1<<cpu for cpu in range(8,14)) for block in ids):continue
            observed.append({'sample':number,'time':sample['time'],'block_ids':list(ids),
                             'owners':[busy[block]['owner'] for block in ids]})
            break
    smi=bool(observed)
    return {'production_v2_ownership_pass':ownership,'smi_six_busy_pass':smi,
            'physical_tcm_verified':ownership and smi,'smi_samples':len(samples),
            'smi_six_busy_samples':len(observed),'first_six_busy':observed[:1],
            'required_block_sets':[list(ids) for ids in sorted(required_sets)],
            'requirement':'production catalog mapping/CPU/TID ownership checks AND one SMI snapshot with all six owned blocks busy'}
