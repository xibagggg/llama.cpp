#!/usr/bin/env python3
"""Export a concrete PP-DocLayoutV3 ONNX graph to native GGML metadata/GGUF."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def export(source, target, pretranspose_gemm=False):
    model = onnx.shape_inference.infer_shapes(onnx.load(source), data_prop=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 4:
        raise ValueError('expected one input and all four DocLayout outputs')
    shapes, types = {}, {}
    type_names = {onnx.TensorProto.FLOAT: 'f32', onnx.TensorProto.INT64: 'i64', onnx.TensorProto.BOOL: 'bool'}
    for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        dims = [d.dim_value for d in item.type.tensor_type.shape.dim]
        if any(d <= 0 for d in dims):
            raise ValueError('unresolved shape: ' + item.name)
        shapes[item.name] = dims
        types[item.name] = type_names[item.type.tensor_type.elem_type]
    if shapes[model.graph.input[0].name] != [1, 3, 800, 800]:
        raise ValueError('expected concrete [1,3,800,800] input')
    names = {}
    def key(name):
        if not name:
            return ''
        if name not in names:
            names[name] = f'v{len(names):05d}'
        return names[name]
    plan = {'version': 1, 'input': key(model.graph.input[0].name), 'input_shape': [1,3,800,800],
            'weights': [], 'nodes': [], 'outputs': []}
    initializers={item.name:item for item in model.graph.initializer}
    eligible=[]
    if pretranspose_gemm:
        for index,node in enumerate(model.graph.node):
            attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type!='Gemm' or attrs.get('transA',0)!=0 or attrs.get('transB',0)!=0:
                continue
            item=initializers.get(node.input[1])
            if item is not None and item.data_type==onnx.TensorProto.FLOAT and len(item.dims)==2:
                eligible.append((index,node))
    packed_names={}
    extra_weights=[]
    occupied={n for node in model.graph.node for n in [*node.input,*node.output]}
    eligible_indices={index for index,_ in eligible}
    for _,node in eligible:
        name=node.input[1]
        if name in packed_names:continue
        all_uses=[(consumer_index,input_index) for consumer_index,consumer in enumerate(model.graph.node)
                  for input_index,n in enumerate(consumer.input) if n==name]
        inplace=all(input_index==1 and consumer_index in eligible_indices for consumer_index,input_index in all_uses)
        # Keep the original layout if the initializer has other consumers.
        packed=name
        if not inplace:
            packed=f'__ocr_gemm_pretranspose_{len(packed_names)}'
            while packed in occupied:packed+='_' 
            occupied.add(packed)
            extra_weights.append((packed,initializers[name]))
        packed_names[name]=packed
    plan['optimizations']={'pretransposed_gemm_nodes':len(eligible),'pretransposed_unique_weights':len(packed_names)}
    writer = gguf.GGUFWriter(target, 'pp-doclayout-v3')
    for tensor_name,item in [(item.name,item) for item in model.graph.initializer]+extra_weights:
        array = numpy_helper.to_array(item)
        if tensor_name!=item.name or packed_names.get(item.name)==item.name:
            array=array.T
        dtype = type_names[item.data_type]
        dims = list(array.shape)
        if array.size == 0 or array.ndim > 4:
            raise ValueError('unsupported initializer: ' + item.name)
        data = np.ascontiguousarray(array.reshape(array.shape or (1,)))
        raw = None
        if dtype == 'bool':
            data = data.astype(np.int8)
            raw = gguf.GGMLQuantizationType.I8
        elif dtype == 'i64':
            raw = gguf.GGMLQuantizationType.I64
        writer.add_tensor(key(tensor_name), data, raw_dtype=raw)
        plan['weights'].append({'id': key(tensor_name), 'shape': dims, 'dtype': dtype})
    for index, node in enumerate(model.graph.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        attrs = {k: v.decode() if isinstance(v,bytes) else v for k,v in attrs.items()}
        inputs=list(node.input)
        if index in eligible_indices:
            inputs[1]=packed_names[inputs[1]]
            attrs['transB']=1
        plan['nodes'].append({'index': index, 'name': node.name, 'op': node.op_type,
            'inputs': [key(n) for n in inputs], 'attrs': attrs,
            'outputs': [{'id': key(n), 'shape': shapes[n], 'dtype': types[n]} for n in node.output]})
    for item in model.graph.output:
        plan['outputs'].append({'name': item.name, 'id': key(item.name), 'shape': shapes[item.name], 'dtype': types[item.name]})
    writer.add_string('ocr.layout.graph', json.dumps(plan, separators=(',',':')))
    writer.add_string('ocr.layout.source_sha256', hashlib.sha256(Path(source).read_bytes()).hexdigest())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    Path(str(target)+'.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'nodes':len(plan['nodes']),'weights':len(plan['weights']),'outputs':plan['outputs']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('target',type=Path)
    parser.add_argument('--pretranspose-gemm',action='store_true',help='Store constant Gemm matrices in compute layout')
    args = parser.parse_args()
    export(args.source,args.target,args.pretranspose_gemm)
