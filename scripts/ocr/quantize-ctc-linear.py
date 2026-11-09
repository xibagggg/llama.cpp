#!/usr/bin/env python3
"""Quantize pretransposed CTC linear weights with explicit zero padding.

Input is the version-2 FP32 pack. Runtime pads activations on K and crops M
before any consumer, so attention/normalization/Softmax keep original shapes.
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
    plan = copy.deepcopy(json.loads(reader.get_field('ocr.ctc.graph').contents()))
    if plan['version'] != 2:
        raise ValueError('expected version 2 FP32 pretransposed CTC graph')
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    descriptors = {item['id']: item for item in plan['weights']}
    if any(t.tensor_type != gguf.GGMLQuantizationType.F32 for t in tensors.values()):
        raise ValueError('input must contain only FP32 weights')
    arrays, originals, records = {}, {}, []
    for node in plan['nodes']:
        if node['op'] != 'MatMul' or not node.get('rhs_pretransposed', False):
            continue
        name = node['inputs'][1]
        desc, tensor = descriptors[name], tensors[name]
        dims = desc['shape']
        if len(dims) != 2 or min(dims) <= 0 or list(tensor.data.shape) != dims:
            raise ValueError('invalid pretransposed weight shape: ' + name)
        if not np.isfinite(tensor.data).all():
            raise ValueError('nonfinite CTC weight: ' + name)
        packed_name = name + '.q8_pad32'
        if packed_name in tensors or packed_name in descriptors:
            raise ValueError('quantized tensor name collision: ' + packed_name)
        padded = [(dim + 31) // 32 * 32 for dim in dims]
        if packed_name not in arrays:
            values = np.zeros(padded, np.float32)
            values[:dims[0], :dims[1]] = tensor.data
            arrays[packed_name] = gguf.quantize(values, gguf.GGMLQuantizationType.Q8_0)
            originals[name] = packed_name
            records.append({'source': name, 'target': packed_name, 'original_shape': dims,
                            'padded_shape': padded, 'original_bytes': tensor.data.nbytes,
                            'quantized_bytes': arrays[packed_name].nbytes})
        node['inputs'][1] = packed_name
        node['rhs_original_shape'] = dims
    if not arrays:
        raise ValueError('no pretransposed constant CTC MatMul weights')
    used = {name for node in plan['nodes'] for name in node['inputs']} | {plan['output']}
    removed = set(originals) - used
    plan['weights'] = [item for item in plan['weights'] if item['id'] not in removed]
    plan['weights'].extend({'id': item['target'], 'shape': item['padded_shape'], 'dtype': 'q8_0'}
                           for item in records)
    plan['version'] = 3
    writer = gguf.GGUFWriter(output, 'ppocr-v5-ctc')
    writer.add_string('ocr.ctc.graph', json.dumps(plan, separators=(',', ':')))
    writer.add_string('ocr.ctc.source_sha256', reader.get_field('ocr.ctc.source_sha256').contents())
    writer.add_string('ocr.ctc.quantized.parent_sha256', hashlib.sha256(source.read_bytes()).hexdigest())
    writer.add_string('ocr.ctc.quantized.scope', 'constant-linear-q8_0-km-pad32; convolution and attention FP32')
    for name, tensor in tensors.items():
        if name not in removed:
            writer.add_tensor(name, np.ascontiguousarray(tensor.data))
    for name, array in arrays.items():
        writer.add_tensor(name, array, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file(); writer.write_kv_data_to_file()
    writer.write_tensors_to_file(); writer.close()
    summary = {'precision': 'q8_0-linear-mixed-gguf', 'quantized_weights': len(arrays),
               'quantized_nodes': sum('rhs_original_shape' in node for node in plan['nodes']),
               'weights': records, 'source_bytes': source.stat().st_size,
               'output_bytes': output.stat().st_size,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    Path(str(output) + '.packing.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack(args.source, args.output)
