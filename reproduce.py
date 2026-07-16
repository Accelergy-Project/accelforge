# %% [markdown]
# # BUG
# 
# Mapper joins successfully, then fails during the detailed run.

# %%

yamlfile = """
arch:
  nodes:
  # ---- Main memory (HBM2) --------------------------------------------------
  - !Hierarchical
    nodes:
    - !Memory
      name: MainMemory
      component_class: HBM2
      size: inf
      leak_power: 0
      area: 0
      tensors: {keep: ~Intermediates, may_keep: All}
      actions:
      - {name: read,  energy: 6.25e-12, throughput: 5277655813324.8, bits_per_action: 1}
      - {name: write, energy: 6.25e-12, throughput: 5277655813324.8, bits_per_action: 1}

  # ---- Cold accelerator (aqfp_partitoned_distributed) ----------------------
  - !Fork
    forked_power_gateable: True
    nodes:
    - !Array
      name: Die
      spatial:
      - {name: xy, fanout: 64, min_usage: 1}
      nodes:
      - !Memory
        name: GlobalBuffer
        component_class: AQFPSRLoopMem
        size: 8*1024*1024*8 # per-die partition, distributed across the 64 dies
        leak_power: 9.350347268903854e-07
        area: 0.0022830435532799997
        spatial: [{name: xy, fanout: 64}]
        tensors:
          keep: ~MainMemory.tensors
          may_keep: All
        actions:
        - {name: read,  energy: 0.0, throughput: 16383999999999.998, bits_per_action: 1}
        - {name: write, energy: 0.0, throughput: 8191999999999.999,  bits_per_action: 1}

      - !Network
        name: NoC
        component_class: AQFP
        leak_power: 4.1112894681832637e-10
        area: 6.188399999999999e-08
        actions:
        - {name: hop, latency: 0, throughput: float("inf"), energy: 0}

    - !Compute
      name: ScalarUnit
      component_class: Dummy
      enabled: len(All) == 2
      leak_power: 0
      area: 0

    - !Memory
      name: LocalBuffer
      component_class: AQFPSRLoopMem
      size: 1*1024*8
      leak_power: 1.0474328899972889e-06
      area: 2.7869183999999997e-07
      tensors:
        keep: All - weight
        tensor_order_options: [[output, weight, input]]
      actions:
      - {name: read,  energy: 0.0, throughput: 32767999999999.996, bits_per_action: 1}
      - {name: write, energy: 0.0, throughput: 16383999999999.998, bits_per_action: 1}

    - !Container
      name: ProcessingElement
      spatial:
      - {name: reuse_input,  fanout: 64,  may_reuse: input,  reuse: input,  min_usage: 1}
      - {name: reuse_output, fanout: 128, may_reuse: output, reuse: output, min_usage: 1}

    - !Memory
      name: Register
      component_class: AQFPLoopReg
      size: weight.bits_per_value if weight else 8
      leak_power: 7.175025249883533e-12
      area: 2.1599999999999998e-10
      tensors: {keep: weight}
      actions:
      - {name: read,  energy: 0.0, throughput: float("inf"), bits_per_action: 1}
      - {name: write, energy: 0.0, throughput: float("inf"), bits_per_action: 1}

    - !Compute
      name: MAC
      component_class: AQFPIntMAC
      enabled: len(All) == 3
      leak_power: 5.087092902167426e-10
      area: 7.657199999999999e-08
      actions:
      - {name: compute, energy: 0.0, throughput: 999999999.9999999}

workload:
  rank_sizes:
    B: 1
    P: 8192
    M: 8192
    H: 96
    E: 128
    F: 128
    D: 96*128

  bits_per_value: {All: 8}
  persistent_tensors: weight - Intermediates

  einsums:
  - einsum: "I[b, m, d] = I_in[b, m, d]"
    is_copy_operation: True
  - "V[b, m, h, e] = I[b, m, d] * WV[h, e, d]"
  - "K[b, m, h, e] = I[b, m, d] * WK[h, e, d]"
  - "Q[b, m, h, e] = I[b, m, d] * WQ[h, e, d]"
  - einsum: "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M: p, h, e]"
    renames: {input: Q}
  - einsum: "AV[b, m, h, f] = QK[b, m, p, h] * V[b, M: p, h, E: f]"
    renames: {input: QK}

renames:
  einsums:
  - name: default
    tensor_accesses:
    - name: input
      source: Inputs & Intermediates if len(All) == 3 else Inputs
      expected_count: 1
    - name: output
      source: Outputs
      expected_count: 1
    - name: weight
      source: ~(input | output)
      expected_count: 1 if len(All) == 3 else 0
"""

# %%
import accelforge as af
# af.set_n_parallel_jobs(1)

open("spec.yaml", "w").write(yamlfile)
spec = af.Spec.from_yaml("spec.yaml")
spec.mapper.metrics = af.Metrics.ENERGY_DELAY_PRODUCT
mappings = spec.map_workload_to_arch()

