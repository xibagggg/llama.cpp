#!/usr/bin/env python3
"""Quantize constant, pretransposed DocLayout Gemm weights for GGML execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def pack(source, target, precision="q8_0"):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve() or target.exists():
        raise FileExistsError("refusing to overwrite a source or existing artifact")
    if precision not in ("q4_0", "q8_0"):
        raise ValueError("unsupported linear precision")
    reader = gguf.GGUFReader(source)
    plan = copy.deepcopy(json.loads(reader.fields["ocr.layout.graph"].contents()))
    if plan.get("version") != 1:
        raise ValueError("unsupported layout plan version")
    weights = {w["id"]: w for w in plan["weights"]}
    tensors = {t.name: t for t in reader.tensors}
    uses, selected = {}, {}
    for index, node in enumerate(plan["nodes"]):
        for position, name in enumerate(node["inputs"]):
            uses.setdefault(name, []).append((index, position))
        if node["op"] != "Gemm":
            continue
        attrs = node["attrs"]
        if (attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1
                or attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1):
            continue
        name = node["inputs"][1]
        w = weights.get(name)
        if (w is None or w["dtype"] != "f32" or len(w["shape"]) != 2
                or min(w["shape"]) <= 0 or w["shape"][1] % 32):
            continue
        if tensors[name].tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError("FP32 descriptor/storage mismatch: " + name)
        selected[index] = name
    if not selected:
        raise ValueError("no supported constant Gemm weights")
    rewritten, arrays, records = {}, {}, []
    output_weights = {o["id"] for o in plan.get("outputs", [])}
    for name in dict.fromkeys(selected.values()):
        old = weights[name]
        original_shape = list(old["shape"])
        # Preserve other consumers and any explicitly exposed original weight.
        inplace = name not in output_weights and all(
            position == 1 and index in selected for index, position in uses[name])
        packed_name = name if inplace else "__ocr_linear_matrix_" + name
        while not inplace and (packed_name in weights or packed_name in arrays):
            packed_name += "_"
        array = np.ascontiguousarray(tensors[name].data.reshape(original_shape), dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite linear weights: " + name)
        arrays[packed_name] = gguf.quantize(array, gguf.GGMLQuantizationType[precision.upper()])
        descriptor = {"id": packed_name, "shape": original_shape, "dtype": precision}
        if inplace:
            weights[name].clear()
            weights[name].update(descriptor)
        else:
            plan["weights"].append(descriptor)
        rewritten[name] = packed_name
        records.append({"source": name, "target": packed_name, "shape": original_shape,
                        "inplace": inplace})
    for index, name in selected.items():
        node = plan["nodes"][index]
        node["inputs"][1] = rewritten[name]
        node["attrs"]["ggml_linear_weight_shape"] = [int(d) for d in tensors[name].shape[::-1]]
    plan.setdefault("optimizations", {}).update(
        quantized_linear_nodes=len(selected), quantized_linear_precision=precision)
    writer = gguf.GGUFWriter(target, "pp-doclayout-v3")
    for item in plan["weights"]:
        name = item["id"]
        if name in arrays:
            array, raw = arrays[name], gguf.GGMLQuantizationType[precision.upper()]
        else:
            tensor = tensors[name]
            array, raw = np.ascontiguousarray(tensor.data), tensor.tensor_type
        writer.add_tensor(name, array, raw_dtype=raw)
    writer.add_string("ocr.layout.graph", json.dumps(plan, separators=(",", ":")))
    writer.add_string("ocr.layout.source_sha256", reader.fields["ocr.layout.source_sha256"].contents())
    writer.add_string("ocr.layout.parent_gguf_sha256", hashlib.sha256(source.read_bytes()).hexdigest())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    Path(str(target) + ".json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    summary = {"precision": precision, "quantized_linear_nodes": len(selected),
               "unique_linear_weights": len(rewritten), "weights": records,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_bytes": source.stat().st_size, "target_bytes": target.stat().st_size,
               "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--precision", choices=["q4_0", "q8_0"], default="q8_0")
    args = parser.parse_args()
    pack(args.source, args.target, args.precision)
