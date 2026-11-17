# Optional real-TCM provenance and v2 ownership verification

This diagnostic is off by default. Enable only with the existing full-pool SPERT
pin and six workers on CPU 8-13:

```sh
export GGML_SPACEMIT_SHARED_MODE=PINNED_TCM
export GGML_SPACEMIT_TCM_OWNERSHIP=1
# Optional: print the worker records after every graph, rather than first/error.
export GGML_SPACEMIT_TCM_DIAGNOSTICS=1
```

The host binds `spine_tcm.h` before any worker starts and rejects an unavailable
runtime, fake/unknown TCM, missing mapping, duplicate pointer, invalid geometry,
or worker count above the fixed eight-lane diagnostic capacity. This capacity
check is not permission to use eight workers on this device. Use six.
`block_info(id).va` reads the existing mappings; this guard never calls
`mem_get`/`mem_free`, creates a DDR substitute, or force-releases a lock. All eight
catalog pointers must be present even if six are used; a missing mapping fails
closed on the host instead of allocating it.

Each worker first pins its SPERT pool, identifies the exact TCM block from its
real shared-buffer pointer, validates the CPU affinity and size, and then calls
`try_wait(id, 0)` on that same Linux TID. CPU affinity alone is insufficient:
blocks 0/1 share mask 0x300, 2/3 share 0xc00, and 4/5 share 0x3000.
The returned pointer, acquired flag, and owner TID must match. New ownership API
failures never throw from an individual worker. Scratch preparation (including
the existing pool pin construction and invalid shared-buffer check) is wrapped
in a stack `optional`/`try` path: in ownership mode its failure is published as
`scratch_unavailable`; with ownership disabled the original exception is
rethrown. Every worker reaches one SPERT barrier and, after a successful
collective barrier, observes one atomic failure flag and skips operators if
any preparation/claim failed.
Successful claims are released on the acquiring worker TID before the pool pin
is returned, including collective abort and exception cleanup. Release is
checked and never retried/forced after a failed or wrong-TID release. A new
foreign claim after successful release is valid.

SPERT remains responsible for propagating a failed/broken barrier or an
operator exception to all lanes. If SPERT returns inconsistent barrier status
or fails to cancel peers, this guard cannot repair that runtime condition.

## Why SMI reported free

The installed SPERT 0.7.1 gets TCM mappings with `mem_get`; it does not invoke the
v2 ownership `mem_try_wait`/`mem_release` callbacks. Thus SMI's v2 lock table can
show free while the process maps and uses physical TCM. Mapping occupancy and
v2 ownership registration are separate facts. The existing PINNED_TCM pool lease
does not establish either physical backing provenance or v2 lock registration.
SPERT also has a `posix_memalign` fallback if `mem_get` returns null, so its
reported 393216-byte worker size alone does not prove real TCM.

Read-only disassembly evidence from the installed riscv64 libraries:

- `/usr/lib/libspert.so.0.7.1`: `initialize_runtime_instance` loads/calls
  `mem_get` at 0xd4c6/0xd4d0. The null path calls `posix_memalign` at 0xd4f2.
  Callback table offsets 640/648 (`mem_try_wait`/`mem_release`) are checked but
  never called by this binary.
- `/usr/lib/libspine_tcm.so.3.0.1`: `spine_tcm_runtime_mem_try_wait` 0x22a0
  forwards the timeout to the v2 helper at 0x3618 (call 0x22e2). The helper
  calls the selected lock's wait callback at 0x366e and records Linux `gettid`
  via syscall 178 at 0x3672-0x3682. `mem_release` verifies that TID.
- The default selected lock table pointer at 0x8178 relocates to 0x7ca0;
  its wait callback at table+16 (0x7cb0) relocates to 0x39fa. That callback
  attempts CAS at 0x3a10, then on busy multiplies timeout by 1000 at 0x3a2e-32.
  Timeout zero reaches one bounded retry, decrements to -1 at 0x3a58, and
  branches to ETIMEDOUT at 0x3a54/0x3a7c-88. It does not futex-wait or yield
  indefinitely. The alternate spin wait callbacks at 0x3872/0x392a likewise
  attempt CAS before a bounded zero-timeout failure. This patch uses zero
  timeout to avoid holding a SPERT core grant while waiting for a v2 block.

These addresses are evidence for the installed 3.0.1 binary, not a portable ABI
promise. Future runtime upgrades need the zero-timeout contract revalidated.

## Portable mock verification

From the llama source root, without SPERT or TCM hardware:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -O2 -pthread \
  tests/ocr/test_tcm_ownership.cpp -o /tmp/test-tcm-ownership
/tmp/test-tcm-ownership
```

The mocks cover unavailable/fake/invalid host metadata, paired CPU masks,
unknown DDR pointers, size and lane bounds, busy/foreign ownership, corrupt
returned pointers, metadata failure, wrong TID, failed/stale releases, immediate
foreign handoff, disabled mode, and a real six-host-thread collective abort.
They validate policy and cross-thread atomic failure propagation. They cannot
verify driver or SPERT barrier behavior.

## Device acceptance after review

Use six AI workers, `parallel=1`, CPU 8-13, and a fresh output directory.
First run a short graph and confirm exact pointer/ID/CPU/TID records report
claimed=1 and released=1 with zero errors. Read-only `spacemit-tcm-smi -i` sampling
should show six claimed blocks during a graph and their release between graphs
and after inference. Never use SMI's force/clear operation. Verify unchanged
tokens/output hashes, then compare full-page timing with instrumentation off
and on in isolated runs. Every-graph log output changes timing and is diagnostic.

This protects only the opted-in llama graph path. Layout and other existing TCM
users that ignore the v2 protocol remain outside this guard. It verifies this
worker's backing and registers ownership; it cannot make an uncooperative
external process obey that protocol. This is a correctness/observability change
and has no demonstrated speed gain yet.
