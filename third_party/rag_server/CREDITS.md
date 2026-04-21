# RAG server credits

The code in this directory was vendored from the
[RLinf project](https://github.com/RLinf/RLinf) to avoid an external runtime
dependency during OpenPARL's WideSearch training.

**Upstream source:** RLinf (commit used as reference when vendored into the
parent `miles` fork: [`5788fa97`](https://github.com/GuanxingLu/miles/commit/5788fa97)).

The specific upstream files correspond to:

  * `build_index.py`  ← `examples/agent/tools/search_local_server_qdrant/build_index.py`
  * `qdrant_encoder.py`  ← `examples/agent/tools/search_local_server_qdrant/qdrant_encoder.py`
  * `local_retrieval_server.py`  ← `examples/agent/tools/search_local_server_qdrant/local_retrieval_server.py`

## Modifications from upstream

Per Apache-2.0 §4(b), we note that the following (minor) changes were made on
top of the upstream files; core retrieval / encoder / index-build logic is
unchanged:

  * Import paths adjusted to work standalone inside
    `third_party/rag_server/` (no `rlinf.*` package imports).
  * Path / config handling adjusted to be relative to this directory rather
    than RLinf's example tree.

## License & attribution

RLinf is distributed under the Apache License, Version 2.0. Each vendored
file retains its original copyright header:

    # Copyright 2025 The RLinf Authors.
    # Licensed under the Apache License, Version 2.0 (the "License");
    # ...

The full Apache-2.0 license text is reproduced in the repository root
[`LICENSE`](../../LICENSE) file, which governs both OpenPARL's own code and
the vendored content in this directory (same license, no additional terms).

The canonical upstream license is available at
<https://github.com/RLinf/RLinf/blob/main/LICENSE>.
