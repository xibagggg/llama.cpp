#!/usr/bin/env python3
"""Export a shape-specialized PP-OCRv5 CTC graph and FP32 weights to GGUF."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def export(model_path, output, batch, width):
    model = onnx.load(model_path)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('expected one CTC input and output')
    signature = model.graph.input[0]
    dimensions = signature.type.tensor_type.shape.dim
    for dim, value in zip(dimensions, (batch, 3, 48, width)):
        dim.dim_value = value
    del model.graph.value_info[:]
    for value in model.graph.output:
        value.type.tensor_type.ClearField('shape')
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    shapes = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if any(d <= 0 for d in shape):
            raise ValueError('unresolved shape: ' + value.name)
        shapes[value.name] = shape
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for name, value in constants.items():
        shapes[name] = list(value.shape)
    # Only integer shape/index programs are evaluated during conversion.
    # Every floating-point inference node is retained for GGML execution.
    folded = set()
    for index, node in enumerate(model.graph.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        args = [constants.get(name) for name in node.input]
        value = None
        if node.op_type == 'Shape':
            value = np.array(shapes[node.input[0]], dtype=np.int64)
        elif args and all(a is not None and a.dtype.kind in 'iu' for a in args):
            if node.op_type == 'Gather':
                value = np.take(args[0], args[1], axis=attrs.get('axis', 0))
            elif node.op_type == 'Slice':
                selectors = [slice(None)] * args[0].ndim
                axes = args[3] if len(args) > 3 else np.arange(len(args[1]))
                steps = args[4] if len(args) > 4 else np.ones(len(args[1]), dtype=np.int64)
                for start, end, axis, step in zip(args[1], args[2], axes, steps):
                    selectors[int(axis)] = slice(int(start), int(end), int(step))
                value = args[0][tuple(selectors)]
            elif node.op_type == 'Concat':
                value = np.concatenate(args, axis=attrs.get('axis', 0))
            elif node.op_type == 'Unsqueeze':
                value = np.expand_dims(args[0], tuple(int(a) for a in args[1]))
        if value is not None:
            constants[node.output[0]] = np.asarray(value)
            folded.add(index)
    names = {}
    def key(name):
        if name not in names:
            names[name] = f'v{len(names):04d}'
        return names[name]
    plan = {'version': 1, 'input': key(signature.name), 'input_shape': [batch, 3, 48, width],
            'output': key(model.graph.output[0].name), 'weights': [], 'nodes': []}
    writer = gguf.GGUFWriter(output, 'ppocr-v5-ctc')
    for name, array in constants.items():
        if array.dtype.kind != 'f':
            continue
        if array.ndim > 4 or array.dtype != np.float32:
            raise ValueError('unsupported weight dtype/rank: ' + name)
        tensor_name = key(name)
        writer.add_tensor(tensor_name, np.ascontiguousarray(array.reshape(array.shape or (1,))))
        plan['weights'].append({'id': tensor_name, 'shape': list(array.shape)})
    supported = {'Conv', 'Add', 'Mul', 'HardSwish', 'HardSigmoid', 'Sigmoid', 'Relu',
                 'GlobalAveragePool', 'AveragePool', 'Reshape', 'Transpose', 'Gather',
                 'LayerNormalization', 'MatMul', 'Softmax', 'Concat', 'Squeeze'}
    for index, node in enumerate(model.graph.node):
        if index in folded:
            continue
        if node.op_type not in supported or len(node.output) != 1:
            raise ValueError('unsupported node: ' + node.op_type + ' ' + node.name)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        record = {'index': index, 'op': node.op_type, 'name': node.name,
                  'inputs': [key(name) for name in node.input], 'output': key(node.output[0]),
                  'shape': shapes[node.output[0]], 'attrs': attrs}
        for i, name in enumerate(node.input):
            if name in constants and constants[name].dtype.kind in 'iu':
                record.setdefault('integer_inputs', {})[str(i)] = constants[name].tolist()
        plan['nodes'].append(record)
    writer.add_string('ocr.ctc.graph', json.dumps(plan, separators=(',', ':')))
    writer.add_string('ocr.ctc.source_sha256', hashlib.sha256(Path(model_path).read_bytes()).hexdigest())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    Path(str(output) + '.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'nodes': len(plan['nodes']), 'folded_shape_nodes': len(folded),
                      'weights': len(plan['weights']), 'input_shape': plan['input_shape'],
                      'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--width', type=int, default=128)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 8 or args.width < 32 or args.width > 4096 or args.width % 32:
        parser.error('batch must be 1..8 and width a multiple of 32 in 32..4096')
    export(args.model, args.output, args.batch, args.width)
