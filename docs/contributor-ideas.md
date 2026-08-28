# Starter contributor ideas

These tasks are intentionally bounded for `good first issue` and `help wanted` tickets.
Selected tasks are now published in the issue tracker. Discuss scope before adding a new
runtime dependency.

## Add a synthetic structure family

Create one deterministic shape family that is not a dot pattern or polygon mosaic. Include
exact support/structure truth, one structured case, one matched negative and a manifest entry.
Acceptance: the generator is seed-stable and the negative does not gain semantic acceptance.

## Improve CLI resource-limit messages

Add actionable examples to errors for file bytes, decoded pixels and aggregate burst pixels.
Acceptance: CLI tests cover each error without exposing a full local path or traceback by
default.

## Add an accessibility presentation preset

Add one overlay palette or texture preset without changing masks, ranking or confidence.
Acceptance: saved artifact names remain deterministic and a test verifies that recovery JSON
is otherwise unchanged.

## Document an optional HEIC decoder design

Research a maintained decoder, its binary supply chain, platform support and security update
policy. This task produces a short design note only; HEIC must not enter the core allowlist
until the dependency and malformed-input behavior are reviewed.

## Add a bounded memory smoke fixture

Create an opt-in script that records wall time and peak resident memory for a generated
single image and burst without committing the large inputs. Acceptance: the script reports
configuration, dimensions and environment and exits nonzero when a caller-provided budget is
exceeded.
