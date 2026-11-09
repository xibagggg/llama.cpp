"""Exercise CTC offline packing with shared weights and dynamic attention."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np

repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('packer', repo / 'scripts/ocr/pack-ctc-linear.py')
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


class PackingTests(unittest.TestCase):
    def fixture(self, path, shared=False, malformed=False):
        weight = np.arange(35, dtype=np.float32).reshape(5, 7) / 9 - 2
        plan = {'version': 1, 'input': 'x', 'output': 'z', 'weights': [
            {'id': 'w', 'shape': [7, 5] if malformed else [5, 7]}], 'nodes': [
            {'op': 'MatMul', 'inputs': ['x', 'w'], 'output': 'y'},
            {'op': 'MatMul', 'inputs': ['y', 'dynamic'], 'output': 'z'}]}
        if shared:
            plan['nodes'].append({'op': 'Add', 'inputs': ['w', 'other'], 'output': 'shared'})
        writer = gguf.GGUFWriter(path, 'ppocr-v5-ctc')
        writer.add_string('ocr.ctc.graph', json.dumps(plan))
        writer.add_string('ocr.ctc.source_sha256', 'f' * 64)
        writer.add_tensor('w', weight)
        writer.write_header_to_file(); writer.write_kv_data_to_file()
        writer.write_tensors_to_file(); writer.close()
        return weight

    def check(self, shared):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'in.gguf', Path(folder) / 'out.gguf'
            weight = self.fixture(source, shared)
            summary = packer.pack(source, output)
            reader = gguf.GGUFReader(output)
            plan = json.loads(reader.get_field('ocr.ctc.graph').contents())
            arrays = {t.name: t.data for t in reader.tensors}
            self.assertEqual(plan['version'], 2)
            np.testing.assert_array_equal(arrays[plan['nodes'][0]['inputs'][1]], weight.T)
            self.assertTrue(plan['nodes'][0]['rhs_pretransposed'])
            self.assertNotIn('rhs_pretransposed', plan['nodes'][1])
            self.assertEqual(summary['replaced_matmul_count'], 1)
            if shared:
                np.testing.assert_array_equal(arrays['w'], weight)
                self.assertEqual(plan['nodes'][2]['inputs'][0], 'w')
            else:
                self.assertNotIn('w', arrays)
            with self.assertRaises(ValueError):
                packer.pack(output, Path(folder) / 'twice.gguf')
            with self.assertRaises(ValueError):
                packer.pack(source, source)
            with self.assertRaises(ValueError):
                packer.pack(source, output)
            arrays.clear(); del reader

    def test_non_square_matrix(self): self.check(False)
    def test_shared_weight(self): self.check(True)

    def test_bad_descriptor(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'in.gguf'
            self.fixture(source, malformed=True)
            with self.assertRaises(ValueError):
                packer.pack(source, Path(folder) / 'out.gguf')
            self.assertFalse((Path(folder) / 'out.gguf').exists())


if __name__ == '__main__':
    unittest.main()
