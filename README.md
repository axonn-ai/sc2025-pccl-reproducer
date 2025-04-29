## Frontier Instructions

### Preparing the environment
The following will create a python virtual environment with all the required dependencies. It will also build the aws-ofi-plugin. The only thing you need 
to change in the script is the PROJ_NAME on line 4.
```bash
bash scripts/frontier/create_python_env.sh
```

### Running
We use the script in `scripts/frontier/run_raw_collectives_benchmark.sh` to launch benchmarking runs. Once again, please change the project name in line 4.

The benchmark script executes the following command:

```bash
SCRIPT="python -u benchmark_raw_collectives/all_gather.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --library pccl --test"
```

To switch from all_gather to another collective (e.g., reduce_scatter), change the script filename in the command:

```bash
benchmark_raw_collectives/all_gather.py → benchmark_raw_collectives/reduce_scatter.py
```

Use the `--library` flag to choose the communication backend - `pccl`, `rccl`, `mpi`

Add the `--test` flag to validate correctness of PCCL operations.
⚠️ Note: This flag is not recommended for large-scale runs as it introduces performance overhead.

Add the `--pccl-recursive-alg` flag to switch from the default ring algorithm to recursive doubling/halving, which is typically faster at scale.
