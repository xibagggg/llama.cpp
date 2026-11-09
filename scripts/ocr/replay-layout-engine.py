"""Replay saved SDK layout tensors through one engine; no rendering or service changes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

SHAPES={'logits':(1,300,25),'pred_boxes':(1,300,4),
        'order_logits':(1,300,300),'out_masks':(1,300,200,200)}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_hash(array):return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def memory_snapshot():
    """Read process memory outside timed calls; absent on non-Linux hosts."""
    path=Path('/proc/self/status')
    if not path.is_file():return {'available':False,'source':'/proc/self/status unavailable'}
    fields={}
    for line in path.read_text().splitlines():
        name,_,value=line.partition(':')
        if name in ('VmRSS','VmHWM','VmSize','VmPeak'):
            amount,unit=value.split()
            if unit!='kB':raise ValueError('unexpected /proc memory unit')
            fields[name]=int(amount)*1024
    return {'available':True,'source':'/proc/self/status','bytes':fields,
            'peak_scope':'process lifetime, including input, load, warmup and execution'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('ggml','onnx'),required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True,help='Exact SDK float32 [1,3,800,800] tensor')
    p.add_argument('--output',type=Path,required=True,help='New directory; existing results are never overwritten')
    p.add_argument('--sdk-dir',type=Path,required=True,help='Directory containing the current ggml_layout.py')
    p.add_argument('--library',type=Path)
    p.add_argument('--backend',choices=('cpu','cpu-sched','spacemit'),default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--warmups',type=int,choices=(0,1),default=1)
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=3)
    p.add_argument('--ai-affinity',default='8;9;10;11;12;13;14;15')
    p.add_argument('--grid-names',type=Path,help='Required for EP: exact exported GridSample node list JSON')
    a=p.parse_args()
    if not 1<=a.threads<=16:p.error('threads must be in 1..16')
    if a.engine=='ggml' and not a.library:p.error('GGML requires --library')
    if a.engine=='onnx' and a.backend=='cpu-sched':p.error('cpu-sched is a GGML backend')
    if a.engine=='onnx' and a.backend=='spacemit' and not a.grid_names:p.error('SpaceMIT EP requires --grid-names')
    for name in ('model','input','sdk_dir'):
        setattr(a,name,getattr(a,name).resolve(strict=True))
    if a.library:a.library=a.library.resolve(strict=True)
    if a.input.stat().st_size!=1*3*800*800*4:p.error('input length is not float32 [1,3,800,800]')
    import numpy as np
    pixels=np.fromfile(a.input,dtype='<f4').reshape(1,3,800,800)
    if not np.isfinite(pixels).all():p.error('input contains nonfinite values')
    input_hash=array_hash(pixels)
    if input_hash!=file_hash(a.input):p.error('input byte order or contents changed')
    sys.path.insert(0,str(a.sdk_dir))
    wrapper=a.sdk_dir/'ggml_layout.py'
    if not wrapper.is_file():p.error('--sdk-dir must contain ggml_layout.py')
    environment={k:v for k,v in os.environ.items() if k.startswith(('GGML_SPACEMIT_','OCR_LAYOUT_','SPACEMIT_EP_'))}
    a.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'engine':a.engine,'backend':a.backend,'threads':a.threads,
        'input':{'path':str(a.input),'sha256':input_hash,'shape':list(pixels.shape),'dtype':'float32'},
        'model':{'path':str(a.model),'sha256':file_hash(a.model),'bytes':a.model.stat().st_size},
        'wrapper_sha256':file_hash(wrapper),'script_sha256':file_hash(__file__),
        'machine':platform.machine(),'python':platform.python_version(),
        'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'environment_before_load':environment,'warmups':a.warmups,'repeats':a.repeats,'records':[],
        'preprocessing_excluded':True,'full_model_outputs_saved':True,
        'performance_comparison_or_hardware_coverage_proven':False,
        'memory_before_load':memory_snapshot()}
    def checkpoint(phase,**details):
        report['phase']={'name':phase,**details}
        temporary=a.output/'report.partial.json.tmp'
        temporary.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temporary.replace(a.output/'report.partial.json')
    session=None
    try:
        if a.engine=='ggml':
            os.environ['OCR_LAYOUT_BACKEND']=a.backend
            from ggml_layout import GGMLLayoutSession
            started=time.perf_counter()
            session=GGMLLayoutSession(a.model,a.library,a.threads,True,True)
            report['load_s']=time.perf_counter()-started
            report['library']={'path':str(a.library),'sha256':file_hash(a.library)}
            report['backend_info']=session.describe_backend()
            report['engine_time_scope']='GGML model.compute only; includes scheduler work inside compute, excludes C API input/output transfer and output finite checks'
            def execute():
                before=session.graph_inference_s
                arrays=session.run(None,{'pixel_values':pixels})
                return dict(zip(SHAPES,arrays)),session.graph_inference_s-before
        else:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=a.threads
            options.inter_op_num_threads=1;options.enable_cpu_mem_arena=False
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers=['CPUExecutionProvider'];provider_options={}
            if a.backend=='spacemit':
                import spacemit_ort
                if 'SpaceMITExecutionProvider' not in ort.get_available_providers():raise RuntimeError('SpaceMIT EP is unavailable')
                names=json.loads(a.grid_names.read_text(encoding='utf-8'))
                if not isinstance(names,list) or len(names)!=18 or any(not isinstance(n,str) or not n for n in names):
                    raise ValueError('grid-names must contain exactly 18 nonempty exported node names')
                os.environ['SPACEMIT_EP_INTRA_THREAD_NUM']=str(a.threads)
                os.environ['SPACEMIT_EP_INTRA_THREAD_AFFINITY']=a.ai_affinity
                options.inter_op_num_threads=a.threads
                provider_options={'SPACEMIT_EP_DISABLE_OP_NAME_FILTER':';'.join(names)}
                providers=[('SpaceMITExecutionProvider',provider_options),'CPUExecutionProvider']
                report['grid_names']={'path':str(a.grid_names.resolve()),'sha256':file_hash(a.grid_names),'names':names}
            else:
                options.add_session_config_entry('session.intra_op.allow_spinning','0')
                options.add_session_config_entry('session.inter_op.allow_spinning','0')
            started=time.perf_counter()
            session=ort.InferenceSession(str(a.model),options,providers=providers)
            report['load_s']=time.perf_counter()-started
            session.disable_fallback()
            expected='SpaceMITExecutionProvider' if a.backend=='spacemit' else 'CPUExecutionProvider'
            if session.get_providers()[0]!=expected:raise RuntimeError('requested provider was not selected')
            inputs=session.get_inputs()
            if len(inputs)!=1 or inputs[0].name!='pixel_values' or inputs[0].shape!=[1,3,800,800] or inputs[0].type!='tensor(float)':
                raise ValueError('ONNX input must accept the identical float32 SDK tensor')
            if {o.name for o in session.get_outputs()}!=set(SHAPES):raise ValueError('all four layout outputs are required')
            report['backend_info']={'providers':session.get_providers(),'provider_options':provider_options,'ort_version':ort.__version__}
            report['engine_time_scope']='ONNX Runtime _sess.run elapsed; includes its output allocation/transfer; not identical to GGML graph-only scope'
            def execute():
                begin=time.perf_counter()
                arrays=session._sess.run(list(SHAPES),{'pixel_values':pixels},None)
                elapsed=time.perf_counter()-begin
                return dict(zip(SHAPES,arrays)),elapsed
        report['memory_after_load']=memory_snapshot()
        report['environment_after_load']={k:v for k,v in os.environ.items() if k.startswith(('GGML_SPACEMIT_','OCR_LAYOUT_','SPACEMIT_EP_'))}
        checkpoint('loaded')
        first=None
        for index in range(a.warmups+a.repeats):
            checkpoint('before_execute',index=index,warmup=index<a.warmups)
            begin=time.perf_counter();outputs,engine_s=execute();wall_s=time.perf_counter()-begin
            hashes={}
            for name,shape in SHAPES.items():
                out=np.asarray(outputs[name])
                if out.shape!=shape or out.dtype!=np.float32 or not out.flags.c_contiguous or not np.isfinite(out).all():
                    raise RuntimeError('invalid full output: '+name)
                hashes[name]=array_hash(out)
            if array_hash(pixels)!=input_hash:raise RuntimeError('input changed')
            row={'index':index,'warmup':index<a.warmups,'engine_s':engine_s,'call_wall_s':wall_s,
                 'output_sha256':hashes,'memory_after_execute':memory_snapshot()}
            report['records'].append(row)
            checkpoint('after_execute',index=index)
            if first is None:first=hashes
            elif hashes!=first:raise RuntimeError('repeat or warmup output drift; timing is not accepted')
            if index==a.warmups:
                for name in SHAPES:outputs[name].tofile(a.output/(name+'.f32'))
            print(json.dumps({'index':index,'warmup':row['warmup'],'engine_s':engine_s,'call_wall_s':wall_s}),flush=True)
        formal=[row for row in report['records'] if not row['warmup']]
        report.update(passed=True,input_preserved=True,full_outputs_repeat_exact=True,
            repeat_stability_verified=len(report['records'])>1,
            mean_engine_s=statistics.mean(row['engine_s'] for row in formal),
            mean_call_wall_s=statistics.mean(row['call_wall_s'] for row in formal))
        if a.engine=='ggml' and any(n=='onnxruntime' or n.startswith('onnxruntime.') for n in sys.modules):
            raise RuntimeError('GGML replay unexpectedly imported ONNX Runtime')
    except BaseException as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if session is not None:
                if a.engine=='ggml':session.close()
                else:
                    session._sess=None
                    session=None;gc.collect()
            report['close_completed']=True
            report['memory_after_close']=memory_snapshot()
        except BaseException as error:
            report.update(passed=False,close_completed=False,close_error=str(error))
            raise
        finally:
            report['phase']={'name':'closed' if report.get('close_completed') else 'close_failed'}
            (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
