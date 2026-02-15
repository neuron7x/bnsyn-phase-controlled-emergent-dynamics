# CLI Reference

Reference for the `bnsyn` command-line interface implemented in `src/bnsyn/cli.py`.

## Command group

`bnsyn` exposes four subcommands:

- `demo`
- `run`
- `dtcheck`
- `sleep-stack`

## `bnsyn demo`

Run a deterministic demo simulation, or launch the interactive dashboard.

**Arguments**

- `--steps` (int, default `2000`)
- `--dt-ms` (float, default `0.1`)
- `--seed` (int, default `42`)
- `--N` (int, default `200`)
- `--interactive` (flag, default `False`)

**Examples**

```bash
bnsyn demo --steps 1000 --seed 42 --N 100
bnsyn demo --interactive
```

## `bnsyn run`

Run an experiment from a YAML configuration.

**Arguments**

- `config` (positional path to YAML file)
- `-o, --output` (optional output JSON path)

**Examples**

```bash
bnsyn run examples/configs/quickstart.yaml
bnsyn run examples/configs/quickstart.yaml -o results/my_experiment.json
```

## `bnsyn dtcheck`

Run the dt vs dt/2 invariance harness.

**Arguments**

- `--steps` (int, default `2000`)
- `--dt-ms` (float, default `0.1`)
- `--dt2-ms` (float, default `0.05`)
- `--seed` (int, default `42`)
- `--N` (int, default `200`)

**Example**

```bash
bnsyn dtcheck --dt-ms 0.1 --dt2-ms 0.05 --steps 2000
```

## `bnsyn sleep-stack`

Run the sleep-stack demo with emergence tracking.

**Arguments**

- `--seed` (int, default `123`)
- `--N` (int, default `64`)
- `--backend` (`reference` or `accelerated`, default `reference`)
- `--steps-wake` (int, default `800`)
- `--steps-sleep` (int, default `600`)
- `--out` (string output directory, default `results/sleep_stack_v1`)

**Examples**

```bash
bnsyn sleep-stack --seed 123 --steps-wake 800 --steps-sleep 600 --out results/demo1
bnsyn sleep-stack --seed 123 --steps-wake 240 --steps-sleep 180 --out results/demo_rc
```

## Related docs

- <a href="../README.md">README.md</a>
- [LEGENDARY_QUICKSTART.md](LEGENDARY_QUICKSTART.md)
- [SPEC.md](SPEC.md)
