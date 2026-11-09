#!/usr/bin/env python3
"""Pack constant group=1 DocLayout convolution weights as [M,K] matrices."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def pack(source, target, precision='f32'):
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve() or target.exists():
        raise FileExistsError('refusing to overwrite a source or existing artifact')
    if precision not in ('f32','q4_0','q8_0'):
        raise ValueError('unsupported matrix precision')
    reader=gguf.GGUFReader(source)
    plan=copy.deepcopy(json.loads(reader.fields['ocr.layout.graph'].contents()))
    weights={w['id']:w for w in plan['weights']}
    original_shapes={name:list(w['shape']) for name,w in weights.items()}
    tensors={t.name:t for t in reader.tensors}
    uses={}
    for index,node in enumerate(plan['nodes']):
        for position,name in enumerate(node['inputs']):
            uses.setdefault(name,[]).append((index,position))
    selected={}
    for index,node in enumerate(plan['nodes']):
        if node['op']!='Conv' or node['attrs'].get('group',1)!=1:
            continue
        name=node['inputs'][1]
        weight=weights.get(name)
        if weight is None or weight['dtype']!='f32' or len(weight['shape'])!=4:
            continue
        m,c,kh,kw=weight['shape']; k=c*kh*kw
        if min(m,c,kh,kw)<=0 or (precision!='f32' and k%32):
            continue
        if node['attrs'].get('kernel_shape')!=[kh,kw]:
            raise ValueError('kernel metadata mismatch')
        selected[index]=name
    rewritten={}
    packed_arrays={}
    for name in dict.fromkeys(selected.values()):
        old=weights[name]; m,c,kh,kw=old['shape']; k=c*kh*kw
        inplace=all(position==1 and index in selected for index,position in uses[name])
        packed_name=name if inplace else '__ocr_conv_matrix_'+name
        while not inplace and (packed_name in weights or packed_name in packed_arrays):
            packed_name+='_' 
        array=np.ascontiguousarray(tensors[name].data.reshape(m,k),dtype=np.float32)
        if precision!='f32':
            array=gguf.quantize(array,gguf.GGMLQuantizationType[precision.upper()])
        descriptor={'id':packed_name,'shape':[m,k],'dtype':precision}
        if inplace:
            weights[name].clear();weights[name].update(descriptor)
        else:
            plan['weights'].append(descriptor)
        rewritten[name]=packed_name
        packed_arrays[packed_name]=array
    for index,name in selected.items():
        node=plan['nodes'][index]
        node['attrs']['ggml_conv_weight_shape']=original_shapes[name]
        node['inputs'][1]=rewritten[name]
    plan.setdefault('optimizations',{}).update(matrix_conv_nodes=len(selected),matrix_conv_precision=precision)
    writer=gguf.GGUFWriter(target,'pp-doclayout-v3')
    for item in plan['weights']:
        name=item['id']
        if name in packed_arrays:
            array=packed_arrays[name]
            raw=gguf.GGMLQuantizationType[precision.upper()] if precision!='f32' else None
        else:
            tensor=tensors[name]
            array=np.ascontiguousarray(tensor.data)
            raw=tensor.tensor_type
        writer.add_tensor(name,array,raw_dtype=raw)
    writer.add_string('ocr.layout.graph',json.dumps(plan,separators=(',',':')))
    writer.add_string('ocr.layout.source_sha256',reader.fields['ocr.layout.source_sha256'].contents())
    writer.add_string('ocr.layout.parent_gguf_sha256',hashlib.sha256(source.read_bytes()).hexdigest())
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    Path(str(target)+'.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    summary={'precision':precision,'matrix_conv_nodes':len(selected),'unique_matrix_weights':len(rewritten),
             'source_bytes':source.stat().st_size,'target_bytes':target.stat().st_size,
             'target_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    print(json.dumps(summary))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('target',type=Path)
    parser.add_argument('--precision',choices=['f32','q4_0','q8_0'],default='f32')
    args=parser.parse_args();pack(args.source,args.target,args.precision)
