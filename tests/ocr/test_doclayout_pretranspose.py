"""Check physical GGUF matrices preserve Gemm, including shared consumers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

repo=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('converter',repo/'scripts/ocr/convert-doclayout-gguf.py')
converter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class PretransposeTests(unittest.TestCase):
    def check_case(self,shared=False,trans_b=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a=np.arange(6,dtype=np.float32).reshape(2,3)-2
            b=np.arange(12,dtype=np.float32).reshape(3,4)-5
            stored=b.T.copy() if trans_b else b.copy()
            bias=np.arange(4,dtype=np.float32)
            names=['logits','pred_boxes','order_logits','out_masks']
            nodes=[helper.make_node('Gemm',['A','B','bias'],['logits'],transB=int(trans_b))]
            for name in names[1:]:
                nodes.append(helper.make_node('Identity',['B' if shared and name=='pred_boxes' else 'logits'],[name]))
            outputs=[helper.make_tensor_value_info(name,TensorProto.FLOAT,
                     list(stored.shape) if shared and name=='pred_boxes' else [2,4]) for name in names]
            graph=helper.make_graph(nodes,'test',
                [helper.make_tensor_value_info('pixel_values',TensorProto.FLOAT,[1,3,800,800])],outputs,
                [numpy_helper.from_array(x,name) for name,x in [('A',a),('B',stored),('bias',bias)]])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
            onnx.save(model,root/'test.onnx')
            converter.export(root/'test.onnx',root/'test.gguf',pretranspose_gemm=True)
            plan=json.loads((root/'test.gguf.json').read_text())
            reader=gguf.GGUFReader(root/'test.gguf')
            weights={tensor.name:tensor.data for tensor in reader.tensors}
            node=plan['nodes'][0]
            packed=weights[node['inputs'][1]]
            right=packed.T if node['attrs'].get('transB',0) else packed
            np.testing.assert_array_equal(a@right+bias,a@b+bias)
            if shared:
                original_id=plan['nodes'][1]['inputs'][0]
                np.testing.assert_array_equal(weights[original_id],stored)
                if not trans_b:self.assertNotEqual(original_id,node['inputs'][1])
            self.assertEqual(plan['optimizations']['pretransposed_gemm_nodes'],0 if trans_b else 1)
            # Close memory maps before TemporaryDirectory cleanup on Windows.
            weights.clear();del packed,right,reader

    def test_non_square_matrix(self):self.check_case()
    def test_shared_weight_preserves_other_consumer(self):self.check_case(shared=True)
    def test_already_transposed_matrix(self):self.check_case(trans_b=True)


if __name__=='__main__':unittest.main()
