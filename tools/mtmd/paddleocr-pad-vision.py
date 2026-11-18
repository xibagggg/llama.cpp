#!/usr/bin/env python3
"""Create an independent PaddleOCR-VL Vision candidate from an F16 GGUF.

Only the 27 FFN down matrices are extended from K4304 to K4352 with zeros.
--quantize applies official gguf Q4_0 quantization to all 164 dense matrices;
the patch convolution and existing F32 tensors retain their original bytes.
This is a precision experiment, not a lossless replacement for F16.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--llama-root', type=Path, required=True)
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_root / 'gguf-py'))
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q, GGUFValueType as V
    from gguf.quants import quantize, dequantize

    args.source = args.source.resolve(strict=True)
    args.output = args.output.resolve()
    manifest_path = args.output.with_suffix('.manifest.json')
    if args.source == args.output or args.output.exists() or manifest_path.exists():
        parser.error('Output and manifest must be new files, separate from the source')
    reader = GGUFReader(str(args.source))
    get = lambda key: reader.fields[key].contents()
    assert get('clip.projector_type') == 'paddleocr'
    assert get('clip.vision.block_count') == 27
    assert get('clip.vision.embedding_length') == 1152
    assert get('clip.vision.feed_forward_length') == 4304
    assert len(reader.tensors) == 443
    assert all(t.tensor_type in (Q.F16, Q.F32) for t in reader.tensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(args.output), str(get('general.architecture')), use_temp_file=True)
    for key, field in reader.fields.items():
        if key.startswith('GGUF.'):
            continue
        if key == 'general.file_type' and args.quantize:
            writer.add_uint32(key, 2)
        else:
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == V.ARRAY else None)
    if args.quantize:
        writer.add_uint32('general.quantization_version', 2)
    records = []
    padded_layers = []
    for tensor in reader.tensors:
        data = np.asarray(tensor.data)
        padded = re.fullmatch(r'v\.blk\.(\d+)\.ffn_down\.weight', tensor.name)
        record = {'name': tensor.name, 'source_shape': [int(v) for v in tensor.shape],
                  'source_type': tensor.tensor_type.name,
                  'source_data_sha256': hashlib.sha256(data.tobytes()).hexdigest()}
        if padded:
            assert tensor.tensor_type == Q.F16 and list(tensor.shape) == [4304, 1152]
            original = data.reshape(1152, 4304)
            data = np.pad(original, ((0, 0), (0, 48)), mode='constant')
            assert data[:, :4304].tobytes() == original.tobytes()
            assert not np.any(data[:, 4304:].view(np.uint16))
            padded_layers.append(int(padded[1]))
        dtype = tensor.tensor_type
        if args.quantize and dtype == Q.F16 and data.ndim == 2:
            assert data.shape[-1] % 32 == 0, tensor.name
            source_f32 = data.astype(np.float32)
            data = quantize(source_f32, Q.Q4_0)
            restored = dequantize(data, Q.Q4_0)
            error = restored - source_f32
            record.update(max_abs_error=float(np.max(np.abs(error))),
                          rmse=float(np.sqrt(np.mean(error.astype(np.float64) ** 2))))
            if padded:
                assert not np.any(restored[:, 4304:]), 'Quantized padding must decode to zero'
            dtype = Q.Q4_0
        writer.add_tensor(tensor.name, data, raw_dtype=dtype)
        record.update(output_type=dtype.name, output_data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                      output_data_bytes=int(data.nbytes), padded_down=bool(padded))
        records.append(record)
    assert sorted(padded_layers) == list(range(27))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    result = GGUFReader(str(args.output))
    assert len(result.tensors) == len(records)
    for tensor, record in zip(result.tensors, records):
        assert tensor.name == record['name'] and tensor.tensor_type.name == record['output_type']
        assert hashlib.sha256(tensor.data.tobytes()).hexdigest() == record['output_data_sha256']
        if record['padded_down']:
            assert list(tensor.shape) == [4352, 1152]
        else:
            assert list(tensor.shape) == record['source_shape']
        if not record['padded_down'] and tensor.tensor_type.name == record['source_type']:
            assert record['output_data_sha256'] == record['source_data_sha256']
    types = dict(Counter(t.tensor_type.name for t in result.tensors))
    assert types == ({'F32': 278, 'Q4_0': 164, 'F16': 1} if args.quantize else {'F32': 278, 'F16': 165})
    manifest = {'source': str(args.source), 'source_sha256': digest(args.source),
                'candidate': str(args.output), 'candidate_sha256': digest(args.output),
                'candidate_bytes': args.output.stat().st_size, 'tool_sha256': digest(Path(__file__)),
                'tensor_count': len(records), 'types': types, 'padded_layers': sorted(padded_layers),
                'padding': '27 down weights: zeros appended only at K4304:4352; activation must pad AFTER GELU',
                'precision': 'Q4_0 dense matrices, original F16 patch and F32 tensors' if args.quantize else 'F16 zero-padding control',
                'lossless_precision_claim': not args.quantize, 'records': records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
