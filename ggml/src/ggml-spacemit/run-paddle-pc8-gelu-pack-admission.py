#!/usr/bin/env python3
"""Sequential six-worker hardware admission. Run only under the board owner lock.

Fresh output directory, exact binary/header hashes and subprocess results are
retained. This does not benchmark, launch services, or modify any environment
outside its children. A failed/timeout run writes a final FAIL manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--raw-only", action="store_true", help="Policy/raw/nonfinite gates only; no graph claim")
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    bindir = Path(args.bin_dir).resolve()
    names = ["test-paddle-pc8-gelu-pack-policy", "test-paddle-pc8-gelu-pack-raw",
             "test-paddle-pc8-gelu-pack-graph"]
    binaries = {n: bindir / n for n in (names[:2] if args.raw_only else names)}
    manifest = {"status": "RUNNING", "started_unix": time.time(), "benchmark": False,
                "workers": 6, "allow_eight": 0, "ai_cores": [8, 9, 10, 11, 12, 13],
                "timeout_per_child": args.timeout, "binaries": {}, "runs": []}
    manifest["graph_requested"] = not args.raw_only
    manifest_path = out / "admission.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def run(label, binary, extra=(), flag="0"):
        env = os.environ.copy()
        env.update({"GGML_SPACEMIT_WORKERS": "6", "GGML_SPACEMIT_ALLOW_EIGHT_WORKERS": "0",
                    "SPACEMIT_PERFER_CORE_ID": "8,9,10,11,12,13", "SPERT_BACKEND": "spacemit-k3",
                    "GGML_SPACEMIT_SHARED_MODE": "PINNED_TCM", "SPERT_SHARED_MODE": "TCM",
                    "GGML_SPACEMIT_PADDLE_PC8_GELU_PACK": flag})
        # Timing/trace from an owner's interactive shell must not contaminate
        # admission or fill unrelated evidence buffers. The graph enables its
        # own bounded log callback; the raw fixture prints every worker identity.
        for key in ("GGML_SPACEMIT_OCR_PC8_PHASE_PATH", "GGML_SPACEMIT_OP_TIMING_PATH"):
            env.pop(key, None)
        command = [str(binaries[binary]), *map(str, extra)]
        t0 = time.monotonic()
        try:
            r = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=False)
            code, output, timeout = r.returncode, r.stdout, False
        except subprocess.TimeoutExpired as e:
            code, output, timeout = None, e.stdout or b"", True
        path = out / (label + ".log")
        path.write_bytes(output)
        text = output.decode("utf-8", errors="replace")
        item = {"label": label, "command": command, "producer_flag": flag, "returncode": code,
                "timeout": timeout, "elapsed_seconds": time.monotonic() - t0,
                "log": str(path), "sha256": sha(path)}
        manifest["runs"].append(item)
        save()
        return code, text

    try:
        for name, path in binaries.items():
            manifest["binaries"][name] = {"path": str(path), "sha256": sha(path)}
        save()
        code, text = run("policy", names[0])
        if code != 0 or "PC8_GELU_PACK_POLICY cases=6192 status=PASS" not in text:
            raise RuntimeError("policy admission failed")
        code, text = run("raw", names[1])
        if code != 0 or "cases=400 workers=6 ai_mask=0x3f00 five_frm=1 exact_token_scale_and_pack=1 PASS" not in text:
            raise RuntimeError("raw numerical admission failed")
        lanes = re.findall(r"PC8_GELU_PACK_TCM M=33 lane=(\d+) cpu=(\d+) vlenb=(\d+) packed_groups=(\d+).*canaries=(\d+)", text)
        if {(int(l), int(c), int(v), int(n), int(g)) for l, c, v, n, g in lanes} != {(i, i + 8, 128, 2 if i < 3 else 1, 1) for i in range(6)} or len(lanes) != 6:
            raise RuntimeError("raw worker identity/TCM evidence incomplete")
        special = []
        for kind in range(4):
            a, at = run("special-%d-reference" % kind, names[1], ["--special-reference", kind])
            b, bt = run("special-%d-candidate" % kind, names[1], ["--special-candidate", kind], "1")
            # Match accepted outputs bitwise (two independent hashes), or
            # require identical native fatal rejection and the original
            # nonfinite absmax assertion. Two arbitrary crashes never pass.
            pattern = r"PC8_GELU_PACK_SPECIAL kind=%d packed_hash=([0-9a-f]+) scale_hash=([0-9a-f]+)" % kind
            am, bm = re.search(pattern, at), re.search(pattern, bt)
            accepted = a == b == 0 and am is not None and bm is not None and am.groups() == bm.groups()
            rejected = (a is not None and a == b and a < 0 and "isfinite(maximum)" in at and "isfinite(maximum)" in bt)
            special.append({"kind": kind, "accepted_identical": accepted, "same_assert_rejection": rejected})
            if not (accepted or rejected):
                raise RuntimeError("nonfinite reference/candidate differs for kind %d" % kind)
        manifest["special_values"] = special
        if args.raw_only:
            manifest["status"] = "PASS"
            manifest["native_graph_validated"] = False
            return 0
        reference = out / "native-off-reference.bin"
        code, text = run("graph-reference", names[2], ["--write-reference", reference])
        if code != 0 or "bitwise_complete=1 PASS" not in text:
            raise RuntimeError("native graph reference incomplete")
        manifest["graph_reference"] = {"path": str(reference), "sha256": sha(reference), "bytes": reference.stat().st_size}
        code, text = run("graph-candidate", names[2], ["--compare-reference", reference], "1")
        if code != 0 or "bitwise_complete=1 PASS" not in text:
            raise RuntimeError("native graph candidate failed")
        manifest["status"] = "PASS"
        manifest["native_graph_validated"] = True
    except Exception as e:
        manifest["status"], manifest["error"] = "FAIL", str(e)
    finally:
        manifest["finished_unix"] = time.time()
        save()
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
