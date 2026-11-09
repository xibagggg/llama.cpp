#!/usr/bin/env python3
"""Move constant FP32 CTC MatMul RHS transposes into an offline GGUF pack.

This preserves arithmetic precision. Dynamic attention MatMuls stay unchanged.
The result uses graph version 2 and requires a compatible CTC executor.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def pack(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('output must be a new file distinct from the source')
    reader = gguf.GGUFReader(source)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    plan = copy.deepcopy(json.loads(reader.get_field('ocr.ctc.graph').contents()))
    if plan['version'] != 1:
        raise ValueError('expected an unpacked version 1 CTC graph')
    descriptors = {item['id']: item for item in plan['weights']}
    replacements, arrays = {}, {}
    for node in plan['nodes']:
        if node['op'] != 'MatMul' or node['inputs'][1] not in descriptors:
            continue
        original = node['inputs'][1]
        descriptor, tensor = descriptors[original], tensors[original]
        if len(descriptor['shape']) != 2 or tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError('constant MatMul RHS must be a two-dimensional FP32 weight')
        if list(tensor.data.shape) != descriptor['shape']:
            raise ValueError('constant MatMul descriptor/storage shape mismatch')
        if original not in replacements:
            name = original + '.ctc_mm_t'
            if name in tensors or name in descriptors:
                raise ValueError('packed tensor name collision: ' + name)
            replacements[original] = name
            arrays[name] = np.ascontiguousarray(tensor.data.T)
        node['inputs'][1] = replacements[original]
        node['rhs_pretransposed'] = True
    if not replacements:
        raise ValueError('no eligible constant MatMul weights')
    used = {name for node in plan['nodes'] for name in node['inputs']}
    used.add(plan['output'])
    # Keep originals if a non-MatMul consumer still needs their original layout.
    removed = {name for name in replacements if name not in used}
    plan['weights'] = [item for item in plan['weights'] if item['id'] not in removed]
    plan['weights'].extend({'id': name, 'shape': list(array.shape)} for name, array in arrays.items())
    plan['version'] = 2
    writer = gguf.GGUFWriter(output, 'ppocr-v5-ctc')
    writer.add_string('ocr.ctc.graph', json.dumps(plan, separators=(',', ':')))
    writer.add_string('ocr.ctc.source_sha256', reader.get_field('ocr.ctc.source_sha256').contents())
    writer.add_string('ocr.ctc.linear.source_gguf_sha256', hashlib.sha256(source.read_bytes()).hexdigest())
    writer.add_string('ocr.ctc.linear.packing', 'constant-rhs-transpose-f32-v1')
    for name, tensor in tensors.items():
        if name in removed:
            continue
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError('only FP32 CTC source tensors are supported')
        writer.add_tensor(name, np.ascontiguousarray(tensor.data))
    for name, array in arrays.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    report = {'source': str(source), 'output': str(output), 'precision': 'FP32 unchanged',
              'packed_weight_count': len(arrays), 'removed_original_weights': len(removed),
              'replaced_matmul_count': sum(n.get('rhs_pretransposed', False) for n in plan['nodes']),
              'constant_rhs_bytes': sum(a.nbytes for a in arrays.values()),
              'dynamic_matmul_count': sum(n['op'] == 'MatMul' and not n.get('rhs_pretransposed', False)
                                          for n in plan['nodes'])}
    Path(str(output) + '.json').write_text(json.dumps(plan, indent=2), encoding='utf8')
    Path(str(output) + '.packing.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack(args.source, args.output)
