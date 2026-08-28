# Security model

## Scope and assumptions

ChromaRecover is a local command-line tool and Python library that may receive malformed or
hostile image files. PNG, JPEG and WebP parsing ultimately relies on Pillow and its native
decoder dependencies. In-memory NumPy and PIL objects are assumed to come from the embedding
Python process, which has already allocated them.

The project does not upload images, create accounts or send telemetry. It is not currently a
multi-user service and does not claim that parsing an untrusted file in the caller's process
is a complete sandbox.

## Implemented controls

- Decoded formats, rather than filename extensions, are restricted to PNG, JPEG and WebP.
- Filesystem inputs are rejected above 100 MiB before `Image.open`.
- Decoded dimensions are rejected above 50 megapixels before RGB conversion.
- Embedded ICC profiles are ignored above 4 MiB before they reach LittleCMS; the status is
  recorded as `oversized_ignored`.
- Burst count is rejected unless it is 2–12 before any frame is decoded.
- Burst dimensions are probed and rejected above 80 megapixels in aggregate before full
  decoding and stacking.
- Output artifact names are fixed by the library; source metadata stores only the basename.
- User document corners must be finite, distinct, close to the image, convex, sufficiently
  large and capable of producing a finite non-singular perspective transform.
- Debug artifacts are opt-in, and the public repository does not contain private fixtures.
- CI workflows declare read-only repository permissions. Dependency and Actions updates are
  monitored by Dependabot, with a scheduled dependency audit.

These are resource budgets, not proof that every accepted input will use the same memory or
runtime. Configuration can raise them deliberately, so service operators must enforce a
separate deployment budget.

## Limits that require process isolation

Python cannot reliably interrupt a native image decoder in the same process. A thread-based
timeout would report a timeout while the native operation continued consuming resources, so
the library does not expose one as a false security guarantee.

A web service, bot or shared ingestion system must therefore:

1. copy input into a non-executable, quota-limited temporary area;
2. run ChromaRecover in an unprivileged disposable process or container;
3. enforce wall-time, address-space and CPU limits outside that process;
4. expose no repository, cloud or long-lived credentials to the worker;
5. validate and re-encode any artifact served to another user;
6. delete inputs and debug outputs according to an explicit retention policy.

Keep Pillow, OpenCV and NumPy updated when security advisories apply. Report suspected
vulnerabilities through the private channel described in [`../SECURITY.md`](../SECURITY.md).
