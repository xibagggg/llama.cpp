# Bounded v50 / v55 experiment reproduction

These are archived board-campaign scripts. Copy them to the existing board
project's `scripts/` directory (not this nested directory) so their project
root calculation and existing `benchmark.py` import resolve correctly.
Test images, models, per-document GT, SDKs and captures are task assets and are
not published. The runners serialize benchmarks and restore the project
service. Passwords are supplied separately and are not stored in these files.

Build the immutable v50 source tag with `scripts/ocr/build-k3.sh`, then create
a new model with `scripts/ocr/pad-ocr-down.py`. Use the named runtime folders
and `MANIFEST.json` identity recorded in the results. Build/archive packaging
scripts contain the original lab paths and should be adapted to a new lab;
they are records, not installation commands. Never overwrite an old runtime
or model when replaying.

The v55 tag is a failed attention candidate, not an upgrade recommendation.
Its numerical edge benchmark prototype was not run and is not included.
