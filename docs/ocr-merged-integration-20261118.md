# OCR integration 2026-11-18

Work root: /home/bianbu/ocr-merged. Frozen revisions: integration-inputs.json.

SDK keeps the newer MiniCPM shared pipeline/layout and adds adapter-selected Paddle bounded text groups, exact RGB mosaics, 520px prose wrap and fallback. MiniCPM retains its independent 704px wrap/midline recipe and table-only DSpark. Paddle optional table bands preserve branch behavior; generic/MiniCPM still reject them because rows/spans are not preserved. Dedicated profiles use six workers. No source branch changes.

Engine retains Paddle PC8 final layout, producer pack, paired-QK and RoPE alongside MiniCPM DSpark, mapped argmax, GDN snapshot/rollback and TCM ownership. Both GELU-to-pack paths and distinct GQA helpers survive. Removed duplicate Vision initialization log after strict metrics detected it.

Read-only upstream reference: spacemit/k3-optimizations 5625bff2bf0617a287ffbb0b5ee095730c9804f7. Family/region bridge registers model families; validates capabilities, shape/layout/KV/mask; prepares immutable weights outside token execution; owns workspace per context. Unsupported preparation rejects before graph emission and retains original graph. TCM/worker leases last only synchronous execution. Future upstream work must adapt OCR families to this interface, isolate immutable weights from GDN/KV/request state, and preserve refusal/fallback tests. This merge is not already ported to the newer region API. No upstream push.

Validation: host build cores0-5/-j4, six AI workers8-13. SDK CTest18/18 PASS. Initial incorrectly parameterized MiniCPM and overlapping-build runs remain diagnostics, excluded from final performance results. Serial ABBA records retain nine metrics, patches, input/model/binary hashes, logs and complete outputs.
