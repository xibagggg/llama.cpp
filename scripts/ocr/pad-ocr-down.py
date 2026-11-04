"""Pad OCR FFN-down K4304->4352 and optionally quantize using ggml C API.

All other tensor bytes and metadata are preserved. Output is experimental and
requires the matching graph adapter. Never overwrite the source model.
"""
import argparse,ctypes,hashlib,json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gguf-py'))
import gguf

def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--ggml-lib',type=Path,required=True);ap.add_argument('--layers',default='all',help='all, none (F16 padding control), or comma-separated layer IDs')
    a=ap.parse_args();assert a.source.resolve()!=a.output.resolve() and not a.output.exists()
    layers=set(range(27)) if a.layers=='all' else set() if a.layers=='none' else set(map(int,a.layers.split(',')))
    assert layers<=set(range(27))
    lib=ctypes.CDLL(str(a.ggml_lib.resolve()))
    quant=lib.ggml_quantize_chunk;quant.restype=ctypes.c_size_t
    quant.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
    reader=gguf.GGUFReader(a.source)
    assert reader.get_field('clip.vision.feed_forward_length').contents()==4304
    writer=gguf.GGUFWriter(a.output,reader.get_field('general.architecture').contents())
    for key,field in reader.fields.items():
        if key.startswith('GGUF.') or key=='general.architecture':continue
        writer.add_key_value(key,field.contents(),field.types[0],field.types[-1] if len(field.types)>1 else None)
    writer.add_string('minicpm_ocr.down_padding','K4304->4352; zeros after activation; experimental')
    writer.add_string('minicpm_ocr.down_q4_layers',','.join(map(str,sorted(layers))))
    changed={};preserved={}
    for t in reader.tensors:
        m=re.fullmatch(r'v\.blk\.(\d+)\.ffn_down\.weight',t.name)
        data=t.data;dtype=t.tensor_type
        if m:
            assert tuple(t.shape)==(4304,1152) and dtype==gguf.GGMLQuantizationType.F16
            original=data.reshape(1152,4304).astype(np.float32)
            padded=np.zeros((1152,4352),np.float32);padded[:,:4304]=original
            if int(m[1]) in layers:
                dtype=gguf.GGMLQuantizationType.Q4_0;data=np.empty((1152,4352//32*18),np.uint8)
                assert quant(int(dtype),padded.ctypes.data,data.ctypes.data,0,1152,4352,None)==data.nbytes
                deq=gguf.dequantize(data,dtype)
                assert np.all(deq[:,4304:]==0),t.name
                error=deq[:,:4304]-original
                rel=float(np.linalg.norm(error.astype(np.float64))/np.linalg.norm(original.astype(np.float64)))
                maximum=float(np.max(np.abs(error)))
            else:
                data=padded.astype(np.float16);rel=maximum=0.0
            changed[t.name]={'shape':[4352,1152],'type':dtype.name,'bytes':data.nbytes,'weight_rel_l2':rel,'weight_max_abs_error':maximum}
        else:preserved[t.name]=hashlib.sha256(data.tobytes()).hexdigest()
        writer.add_tensor(t.name,data,raw_dtype=dtype)
    assert len(changed)==27
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file(progress=False);writer.close()
    check=gguf.GGUFReader(a.output)
    for t in check.tensors:
        if t.name in preserved:assert hashlib.sha256(t.data.tobytes()).hexdigest()==preserved[t.name]
        else:assert list(t.shape)==[4352,1152]
    manifest={'source':str(a.source),'source_sha256':hashlib.file_digest(a.source.open('rb'),'sha256').hexdigest(),
              'output':str(a.output),'output_sha256':hashlib.file_digest(a.output.open('rb'),'sha256').hexdigest(),
              'official_ggml_lib':str(a.ggml_lib),'official_ggml_lib_sha256':hashlib.file_digest(a.ggml_lib.open('rb'),'sha256').hexdigest(),
              'q4_layers':sorted(layers),'changed':changed,'preserved_tensor_count':len(preserved),'all_unchanged_tensors_verified':True,'output_bytes':a.output.stat().st_size}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'output':str(a.output),'q4_down_layers':len(layers),'bytes':manifest['output_bytes']}))
if __name__=='__main__':main()
