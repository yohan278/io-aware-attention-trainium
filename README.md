# Lab 4 + CS217 Project: IO-Aware Attention on Trainium

## What this repo already contains (concise)

This repository is a timing-clean Lab 4 system-integration design with a partitioned accelerator and interconnect.

- `GBPartition`: global buffer side (stores/serves data and coordinates execution)
- `PEPartition`: compute side (one or more processing partitions)
- `AxiSplitter`: host-facing AXI router that maps requests to GB or PE partitions
- `DataBus`:
  - `GBSend`: broadcast path from GB to PEs
  - `GBRecv`: reduction/collection path from PEs back to GB
  - `PEStart` / `PEDone`: start/done synchronization network

Core source is in `src/`, HLS flow in `hls/`, FPGA integration in `design_top/`, and reports in `reports/`.

## Project thesis (simple version)

Our core claim is:

> On Trainium-like accelerators, attention performance is often limited by data movement and interconnect behavior (buffer pressure, port contention, and cross-partition traffic), not just FLOPs.

We use this Lab 4 design as a chiplet/multi-die proxy:

- each `PEPartition` represents a local compute island (die partition)
- `GBPartition` acts as a global coordinator/reducer
- `GBRecv` + interconnect paths represent communication bottlenecks

If we exchange compact reduction state instead of large intermediates, we should keep exactness while reducing communication pressure.

## End-to-end data movement

At a high level, the flow is:

1. Host configures GB/PE partitions through AXI (`AxiSplitter` routes by address).
2. `GBPartition` streams/broadcasts work to PE partitions (`GBSend` path).
3. Each `PEPartition` computes local partial results on its shard.
4. PEs return partial results/reduction state (`GBRecv` path).
5. `GBPartition` completes global reduction and emits done/interrupt.

For the attention chiplet proxy specifically:

- `K` and `V` are statically sharded across partitions.
- each partition computes local online-softmax state `(m, l, o_partial)`.
- the global stage merges these states to produce exact attention output.

This is the IO-aware idea in one line: move compact state, not full per-key intermediate tensors.

## How we prove the thesis

We use staged experiments with measurable artifacts.

### Stage 0: fast correctness proof (already implemented)

Run:

```bash
python3 chiplet_proxy_poc.py --seq-len 128 --d-model 32 --d-value 32
```

This script compares:

- baseline exact SDPA
- exact two-partition SDPA with online-softmax state merge

It reports:

- max absolute output difference (correctness)
- exchanged words for state-only communication
- naive communication baselines (score-only, score+value)
- ratio vs state-only exchange

### Stage 1: SystemC mapping in this repo

Map the same reduction pattern into:

- `PEPartition` for local shard compute
- `GBRecv` for aggregation path
- `GBPartition` for final merge/coordination

Then sweep sequence length and tile size to identify when throughput degrades due to communication.

## How we measure interconnect clearly

Interconnect is treated as a first-class metric, not a side note.

For each run, collect:

- **Latency/cycles**: end-to-end completion time
- **Throughput**: useful work completed per unit time
- **Traffic volume**: words/bytes sent on communication paths
- **Contention/backpressure**: queue pressure or stalls at key boundaries
- **Communication efficiency**: state-only exchange vs naive baselines

Primary comparisons:

1. monolithic baseline vs partitioned execution
2. naive partition communication vs state-only online-softmax communication
3. scaling sensitivity across sequence length / tile size

Success criteria:

- output is exact (or numerically equivalent within tolerance)
- communication volume drops significantly
- runtime scaling improves or remains stable as sequence length grows

## Quick commands

### SystemC and HLS/RTL

1. SystemC tests:

```bash
python3 test.py --action systemc_sim
```

2. HLS RTL generation and sim:

```bash
python3 test.py --action rtl_sim
```

### AWS F2 flow

If `hw_sim` fails with a testbench error, pin `aws-fpga` to:
`2f1f343259dcb794adc596a76c31593280fd7c7b`

Setup:

```bash
cd ~/aws-fpga
source hdk_setup.sh
source sdk_setup.sh
cd [path-to-lab4-repo]/design_top
source setup.sh
```

Run hardware simulation:

```bash
make hw_sim
```

Build/program FPGA:

```bash
cd design_top
make fpga_build
make generate_afi
make check_afi_available
make program_fpga
make run_fpga_test
```

## Timing note from Lab 4 solution

- HLS closed at 250 MHz to avoid long buffer paths that appeared at 125 MHz closure.
- `clk_main_a0` is divided to provide 125 MHz to components with synchronization primitives.

Files changed from Lab 4 release repo:

1. `design_top/design/design_top.sv`
2. `design_top/build/constraints/cl_timing_user.xdc`
3. `design_top/build/scripts/synth_design_top.tcl`

## References

- `ac_datatypes_ref.pdf`
- `ac_math_ref.pdf`
- `connections-guide.pdf`
- `catapult_useref.pdf`
- `https://nvlabs.github.io/matchlib`
# Lab 4: System Integration and Interconnect

## Table of Contents
- [1. Introduction](#1-introduction)
  - [File Structure](#file-structure)
- [2. Architecture Overview](#2-architecture-overview)
- [3. SystemC test and HLS to RTL](#3-systemc-test-and-hls-to-rtl)
- [4. FPGA Implementation](#4-fpga-implementation)
- [5. How was Timing fixed?](#5-how-was-timing-fixed)
- [6. Documentation](#6-documentation)


## 1. Introduction 
This repository contains the solution to Lab 4 with a timing-clean AWS environment. 

### File Structure

The repository is organized into the following directories:

-   `src/`: Contains the SystemC source code for the accelerator. This is where you will spend most of your time for this lab.
    -   `src/include/`: Shared header files, including definitions for AXI signals (`AxiSpec.h`) and other specifications.
    -   `src/DataBus/`: Modules for handling data and control signal distribution between the GB and PEs.
    -   `src/Top/GBPartition/`: The Global Buffer partition, which includes the `GBModule` and its sub-modules.
    -   `src/Top/PEPartition/`: The Processing Element partition, which includes the `PEModule` and its sub-modules.
    -   `src/Top/`: The top-level module that integrates the GB and PE partitions.
-   `hls/`: Contains the scripts and Makefiles for running the High-Level Synthesis (HLS) flow. The structure mirrors the `src` directory.
-   `design_top/`: Contains the files for building the design for the AWS F2 FPGA, including Verilog/SystemVerilog files, constraints, and build scripts.
    -   `design_top/design/`: Synthesized RTL files and other design sources.
    -   `design_top/verif/`: Verification environment for the RTL design.
    -   `design_top/build/`: Scripts and constraints for synthesis and implementation.
-   `scripts/`: Utility scripts for the project.
    -   `scripts/hls/`: Scripts for the HLS flow.
    -   `scripts/aws/`: Scripts for interacting with AWS.
-   `docs/`: Contains documentation for the project, including setup guides.
-   `reports/`: Directory for storing reports from HLS and FPGA builds.

## 2. Architecture Overview

The top-level design consists of three main components:

-   **GBPartition**: The Global Buffer partition, which is responsible for storing weights and activations. It contains the `GBModule`.
-   **PEPartition**: The Processing Element partition. The design instantiates multiple PEs. Each partition contains a `PEModule`.
-   **AxiSplitter**: An AXI4 interconnect that routes AXI transactions from the host to the appropriate partition (GB or one of the PEs) based on the address.
-   **DataBus**: A set of modules that manage the broadcasting of data from the GB to all PEs (`GBSend`), the collection of results from PEs back to the GB (`GBRecv`), and the distribution/aggregation of control signals (`PEStart`, `PEDone`).

## 3. SystemC test and HLS to RTL
1. SystemC test - `python3 test.py --action systemc_sim`
2. RTL generation and sim - `python3 test.py --action rtl_sim`

## 4. FPGA Implementation

**Disclaimer:** If your `hw_sim` fails with a testbench error, please check out the following commit for the `aws-fpga` repo before sourcing `hdk_setup.sh`: `git checkout 2f1f343259dcb794adc596a76c31593280fd7c7b`

Clone your lab repository to AWS F2 and set up the environment. This should be similar to previous labs:

```bash
# SSH into AWS F2 instance

# Source AWS F2 environment
cd ~/aws-fpga
source hdk_setup.sh
source sdk_setup.sh

# Move to design_top folder
cd [path-to-lab4-repo]/design_top
source setup.sh
```

Run hardware simulation on AWS F2 for the synthesized RTL:

```bash
# Run RTL simulation
make hw_sim
```

Build the FPGA bitstream, generate the AFI, and program the FPGA:

```bash
cd design_top

make fpga_build          # should take about 2.5 hours
make generate_afi

# Wait for AFI to become available
make check_afi_available

# Once available
make program_fpga
make run_fpga_test
```
### 5. How was Timing fixed?

Credits to @jadbitar for the timing-fixed solution code.

1. HLS closed at 250MHz to prevent long buffer paths in RTL when closed at 125MHz
2. The AWS clock `clk_main_a0` is put through a clock divider to supply 125MHz to all the components. Appropriate synchronization primitives were used.

Files changed from Lab 4 release repo
1. `design_top/design/design_top.sv`
2. `design_top/build/constraints/cl_timing_user.xdc`
3. `design_top/build/scripts/synth_design_top.tcl`

## 6. Documentation

Refer to the following documentation for SystemC and MatchLib, found on both Canvas and in `/cad/mentor/2024.2_1/Mgc_home/shared/pdfdocs`:

- `ac_datatypes_ref.pdf`: Algorithmic C datatypes reference manual. Specifically, see `ac_fixed` and `ac_float` classes.
- `ac_math_ref.pdf`: Algorithmic C math library reference manual. Specifically, see functions for power, reciprocal, and square root.
- `connections-guide.pdf`: Documentation for the Connections library, including detailed information on `Push`/`Pop` semantics and coding best practices.
- `catapult_useref.pdf`: User reference manual for Catapult HLS, including pragmas and synthesis directives. 
- `https://nvlabs.github.io/matchlib`: MatchLib online documentation, including component reference and tutorials.

## 7. 30-Minute Multi-Die Proxy PoC

To prototype your CS217 "chiplet / multi-island" idea quickly, run:

```bash
python3 chiplet_proxy_poc.py --seq-len 128 --d-model 32 --d-value 32
```

What it demonstrates:
- Baseline exact SDPA output.
- Two-partition exact SDPA where `K/V` are statically sharded (proxy for two dies).
- Only online-softmax state is exchanged across partitions (`m`, `l`, partial `o`) rather than full logits.

Why this maps to Lab 4:
- `GBPartition` can model global coordinator/state reduction.
- `PEPartition` instances can model local dies processing their shard.
- `GBRecv`/interconnect models reduction traffic and contention.

Use this script as your first correctness artifact before implementing the same reduction pattern in SystemC.