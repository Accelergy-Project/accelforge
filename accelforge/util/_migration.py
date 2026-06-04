LATENCY_TO_THROUGHPUT_MIGRATION = """
AccelForge has migrated from `latency` to `throughput` to measure action timing. To
migrate, please change all `latency` expressions to `throughput` value being the
reciprocal of the latency. Additionally, `latency_scale` should be changed to
`throughput_scale`. Finally, `total_latency` is still present, but supported expressions
have changed; please refer to its docstring for more information.
"""
