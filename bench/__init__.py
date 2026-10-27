"""Latency benchmarks for the live voice room.

Not tests. The pytest suite is offline and deterministic by design; these probes
deliberately hit the real providers (ElevenLabs, Cerebras) because the numbers we
care about *are* the network. Run them explicitly, record the result, and compare
runs with :mod:`bench.compare` before believing an optimisation worked.
"""
