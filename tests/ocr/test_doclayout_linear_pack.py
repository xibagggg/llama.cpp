"""Check packing with shared/visible weights, skipped shapes and existing Q8 tensors."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

import gguf
import numpy as np

repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("linear_pack", repo / "scripts/ocr/pack-doclayout-linear.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def check():
    with tempfile.TemporaryDirectory(prefix="ocr-linear-pack-") as directory:
        stage = Path(directory)
        source = stage / "source.gguf"
        data = {
            "plain": np.arange(64, dtype=np.float32).reshape(2, 32) / 64,
            "shared": np.arange(96, dtype=np.float32).reshape(3, 32) / 96,
            "visible": np.eye(32, dtype=np.float32),
            "short": np.ones((3, 4), np.float32),
            "not_transposed": np.ones((32, 2), np.float32),
        }
        packed = gguf.quantize(np.ones((2, 32), np.float32), gguf.GGMLQuantizationType.Q8_0)
        plan = {"version": 1, "weights": [{"id": n, "dtype": "f32", "shape": list(a.shape)} for n, a in data.items()],
                "nodes": [], "outputs": [{"id": "visible"}]}
        plan["weights"].append({"id": "conv", "dtype": "q8_0", "shape": [2, 32]})
        for name in data:
            plan["nodes"].append({"op": "Gemm", "inputs": ["x", name],
                "attrs": {"transB": int(name != "not_transposed")}})
        plan["nodes"].append({"op": "Add", "inputs": ["shared", "z"], "attrs": {}})
        plan["nodes"].append({"op": "Gemm", "inputs": ["x", "plain"], "attrs": {"transB": 1}})
        writer = gguf.GGUFWriter(source, "pp-doclayout-v3")
        for name, array in data.items():
            writer.add_tensor(name, array)
        writer.add_tensor("conv", packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        writer.add_string("ocr.layout.graph", json.dumps(plan))
        writer.add_string("ocr.layout.source_sha256", "synthetic-test")
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        source_bytes = source.read_bytes()
        for precision in ("q4_0", "q8_0"):
            target = stage / (precision + ".gguf")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = module.pack(source, target, precision)
            assert summary["quantized_linear_nodes"] == 4 and summary["unique_linear_weights"] == 3
            reader = gguf.GGUFReader(target)
            actual = json.loads(reader.fields["ocr.layout.graph"].contents())
            tensors = {t.name: t for t in reader.tensors}
            for name in ("shared", "visible", "short", "not_transposed"):
                assert tensors[name].tensor_type == gguf.GGMLQuantizationType.F32
                assert tensors[name].data.tobytes() == data[name].tobytes()
            assert tensors["conv"].data.tobytes() == packed.tobytes()
            for node in actual["nodes"]:
                if node["op"] == "Gemm" and "ggml_linear_weight_shape" in node["attrs"]:
                    t = tensors[node["inputs"][1]]
                    assert t.tensor_type == gguf.GGMLQuantizationType[precision.upper()]
                    assert [int(d) for d in t.shape[::-1]] == node["attrs"]["ggml_linear_weight_shape"]
            for name in (source, target):
                try:
                    module.pack(source, name, precision)
                    raise AssertionError("overwriting allowed")
                except FileExistsError:
                    pass
        assert source.read_bytes() == source_bytes
    print(json.dumps({"passed": True, "precisions": 2, "shared_and_visible_weights_preserved": True,
        "existing_quantized_convolution_preserved": True, "source_preserved": True, "overwrite_rejected": True}))


if __name__ == "__main__":
    check()
