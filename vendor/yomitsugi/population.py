"""Snapshot population dynamics and explicit contact/resource diagnostics."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import sys
import time
import numpy as np
from scipy.stats import t as student_t
from .core import Chemistry, Kernel, standard_chemistry


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hash():
    digest = hashlib.sha256()
    runtime = {"python": sys.version, "platform": sys.platform, "numpy": np.__version__,
               "scipy": importlib.metadata.version("scipy"), "rng": "PCG64"}
    digest.update(json.dumps(runtime, sort_keys=True).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=None if compact else 2,
                               separators=(",", ":") if compact else None, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    path, rows = Path(path), list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True, slots=True)
class Individual:
    id: int
    b: int
    g: tuple[int, ...]
    p: tuple[int, ...]
    origin: str = "A"
    pool: int = 0
    reader_parent_id: int | None = None
    template_parent_id: int | None = None


@dataclass(slots=True)
class PopulationState:
    individuals: list[Individual]
    resource: int
    sweep: int = 0
    next_id: int = 0
    cumulative: dict = field(default_factory=lambda: {"births": 0, "deaths": 0, "read_failed": 0, "resource_blocked": 0, "attempted": 0, "overflow": 0})
    extinction_time: int | None = None
    pool_resources: dict[int, int] | None = None


def replicate_seed(base_seed, repeat_id):
    """Independent of worker count, dispatch order, and condition/mode."""
    return int(np.random.SeedSequence([int(base_seed), int(repeat_id)]).generate_state(1, dtype=np.uint64)[0])


def _split_integer(total, pool):
    return total // 2 + (pool == 0) * (total % 2)


@lru_cache(maxsize=65536)
def _observation_keys(b, g, p):
    """Formatting is observational: cache immutable states, never individual IDs."""
    template = f"{b}:{','.join(map(str, g))}"
    reader = ",".join(map(str, p))
    return template, reader, f"{template}|{reader}"


@lru_cache(maxsize=65536)
def _word_json(word):
    return json.dumps(word)


def sweep(state, kernel, rng, config, event_sink=None):
    """One sweep. Labels only select pools in explicitly restricted controls."""
    J, cap, death = int(config["J"]), int(config["Smax"]), float(config["d"])
    if J < 0 or cap < 0 or not 0 <= death <= 1:
        raise ValueError("Invalid resource or death parameters")
    snapshot = state.individuals
    n, s_before = len(snapshot), state.resource
    mode = config.get("contact_mode", "mixed")
    restricted = mode in {"pooled_shared", "pooled_separate"}
    split = mode == "pooled_separate"
    if split:
        if state.pool_resources is None:
            state.pool_resources = {k: _split_integer(s_before, k) for k in (0, 1)}
        overflow = 0
        for k in (0, 1):
            offered, pool_cap = _split_integer(J, k), _split_integer(cap, k)
            overflow += max(0, state.pool_resources[k] + offered - pool_cap)
            state.pool_resources[k] = min(pool_cap, state.pool_resources[k] + offered)
        state.resource = sum(state.pool_resources.values())
    else:
        overflow = max(0, state.resource + J - cap)
        state.resource = min(cap, state.resource + J)
    pools = {k: [v for v in snapshot if v.pool == k] for k in (0, 1)} if restricted else {}
    children = []
    blocked = failed = attempted = 0
    directional = Counter()
    for position, index in enumerate(rng.permutation(n)):
        actor = snapshot[int(index)]
        if not split and state.resource == 0:
            blocked += n - position
            break
        if split and state.pool_resources[actor.pool] == 0:
            blocked += 1
            continue
        targets = pools[actor.pool] if restricted else snapshot
        target = targets[int(rng.integers(len(targets)))]
        attempted += 1
        output = kernel.read_word(actor.p, target.b, target.g, rng)
        if output is None:
            failed += 1
            continue
        if config.get("parent_copy", False):
            output = actor.p
            if len(output) != len(target.g):
                raise ValueError("Parent-copy control requires matching word lengths")
        state.resource -= 1
        if split:
            state.pool_resources[actor.pool] -= 1
        child = Individual(state.next_id, target.b, target.g, tuple(output), target.origin, target.pool, actor.id, target.id)
        state.next_id += 1
        children.append(child)
        directional[f"{actor.origin}->{target.origin}"] += 1
        if event_sink is not None:
            event_sink({"sweep": state.sweep + 1, "child_id": child.id,
                        "reader_parent_id": actor.id, "template_parent_id": target.id,
                        "reader_origin": actor.origin, "template_origin": target.origin,
                        "b": child.b, "g": _word_json(child.g), "p": _word_json(child.p)})
    candidates = snapshot + children
    survive = rng.random(len(candidates)) >= death
    state.individuals = [v for v, keep in zip(candidates, survive) if keep]
    deaths, births = len(candidates) - len(state.individuals), len(children)
    state.sweep += 1
    if not state.individuals and state.extinction_time is None:
        state.extinction_time = state.sweep
    assert state.resource == s_before + J - overflow - births
    assert len(state.individuals) == n + births - deaths
    assert births <= min(n, s_before + J - overflow)
    assert attempted + blocked == n
    assert births + failed == attempted
    increments = dict(births=births, deaths=deaths, read_failed=failed,
                      resource_blocked=blocked, attempted=attempted, overflow=overflow)
    for name, value in increments.items():
        state.cumulative[name] = state.cumulative.get(name, 0) + int(value)
    # Count numeric state once, then format only distinct occupied states. This
    # preserves every count and ordering in the saved sorted JSON dictionaries.
    occupied = Counter((v.b, v.g, v.p, v.origin) for v in state.individuals)
    template_counts, origin_counts, reader_counts, joint_counts = Counter(), Counter(), Counter(), Counter()
    for (b, g, p, origin), count in occupied.items():
        template, reader, joint = _observation_keys(b, g, p)
        template_counts[template] += count
        origin_counts[origin] += count
        reader_counts[reader] += count
        joint_counts[joint] += count
    return dict(sweep=state.sweep, N_before=n, N=len(state.individuals), S_before=s_before,
                S=state.resource, offered=J, **increments,
                origin_counts=json.dumps(dict(origin_counts), sort_keys=True),
                template_counts=json.dumps(dict(template_counts), sort_keys=True),
                reader_counts=json.dumps(dict(reader_counts), sort_keys=True),
                template_reader_counts=json.dumps(dict(joint_counts), sort_keys=True),
                direction_births=json.dumps(dict(directional), sort_keys=True),
                distinct_readers=len(reader_counts), extinct=not bool(state.individuals))


def save_checkpoint(path, state, rng, config_hash, code_hash, history=None, birth_events=None):
    atomic_json(path, {"state": asdict(state), "rng_state": rng.bit_generator.state,
                      "config_hash": config_hash, "code_hash": code_hash,
                      "history": history or [], "birth_events": birth_events or []}, compact=True)


def load_checkpoint(path, config_hash, code_hash):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value["config_hash"] != config_hash or value["code_hash"] != code_hash:
        raise ValueError("Checkpoint configuration/code mismatch: use a separate run directory")
    raw = value["state"]
    raw["individuals"] = [Individual(**{**v, "g": tuple(v["g"]), "p": tuple(v["p"])}) for v in raw["individuals"]]
    if raw.get("pool_resources") is not None:
        raw["pool_resources"] = {int(k): v for k, v in raw["pool_resources"].items()}
    rng = np.random.default_rng()
    rng.bit_generator.state = value["rng_state"]
    return PopulationState(**raw), rng, value["history"], value["birth_events"]


def _chemistry(raw):
    return Chemistry(**{name: {int(k): float(v) for k, v in values.items()} for name, values in raw.items()})


def default_config(profile="smoke"):
    chemistry = {"backbone_features": {0: 0.0}, "symbol_features": {0: 0.0, 1: 1.0},
                 "residue_key_features": {0: 0.0, 1: 1.0}, "residue_backbone_features": {0: 0.0, 1: 0.0}}
    types = [{"name": "identity", "b": 0, "g": [0, 0, 1, 1], "p": [0, 0, 1, 1]},
             {"name": "swap", "b": 0, "g": [1, 0, 0, 1], "p": [0, 1, 1, 0]}]
    return dict(sweeps=1000 if profile == "main" else 50, repeats=20 if profile == "main" else 3,
                initial_population=60, S0=60, Smax=120, J=30, d=0.2,
                base_seed=2026102405, modes=["single_a", "single_b", "mixed", "pooled_shared", "pooled_separate"],
                conditions=[dict(name=f"identity_swap_sigma_{sigma}", sigma=sigma, chemistry=chemistry, types=types) for sigma in (0.0, 0.5)],
                wall_seconds=86400, max_population=100000, max_trials=100000000,
                memory_gb=20, disk_gb=5, birth_log_limit=2000, checkpoint_seconds=240,
                sample_every=1, parent_copy=False)


def _initial_state(config, condition, mode):
    population, n = [], config["initial_population"]
    for i in range(n):
        k = 0 if mode == "single_a" else 1 if mode == "single_b" else int(i >= n // 2)
        kind = condition["types"][k]
        population.append(Individual(i, int(kind["b"]), tuple(kind["g"]), tuple(kind["p"]), kind["name"], k))
    return PopulationState(population, int(config["S0"]), next_id=n)


def _rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _run_task(task):
    config, condition, mode, repeat_id, output_dir, config_hash, code_hash, resume, workers, deadline, task_count = task
    started = time.monotonic()
    root = Path(output_dir)
    task_dir = root / f"{condition['name']}__{mode}__r{repeat_id:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, summary_path = task_dir / "checkpoint.json", task_dir / "summary.json"
    if resume and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing["termination_reason"] == "normal_completion":
            return existing
    seed = replicate_seed(config["base_seed"], repeat_id)
    kernel = Kernel(_chemistry(condition["chemistry"]), condition["sigma"])
    if resume and checkpoint.exists():
        state, rng, history, events = load_checkpoint(checkpoint, config_hash, code_hash)
    else:
        state, rng, history, events = _initial_state(config, condition, mode), np.random.default_rng(seed), [], []
    last_checkpoint = time.monotonic()
    settings, reason = dict(config, contact_mode=mode), "normal_completion"
    event_limit = int(config.get("birth_log_limit", 2000))
    def receive_event(event):
        if len(events) < event_limit:
            events.append(event)
    metadata = dict(run_id=root.name, config_hash=config_hash, code_hash=code_hash,
                    stage="population", repeat_id=repeat_id, seed=seed,
                    condition=condition["name"], mode=mode, observation_length=config["sweeps"])
    try:
        while state.sweep < config["sweeps"]:
            if time.monotonic() >= deadline:
                reason = "time_limit"; break
            if len(state.individuals) >= config["max_population"]:
                reason = "population_limit"; break
            if state.cumulative["attempted"] >= config["max_trials"]:
                reason = "trial_limit"; break
            if _rss_bytes() * workers > config["memory_gb"] * 1024 ** 3:
                reason = "memory_limit"; break
            sink = receive_event if len(events) < event_limit else None
            history.append(sweep(state, kernel, rng, settings, sink))
            if time.monotonic() - last_checkpoint >= min(240, config["checkpoint_seconds"]):
                save_checkpoint(checkpoint, state, rng, config_hash, code_hash, history, events)
                # Each task receives an equal disk allowance. The factor 3
                # reserves checkpoint, trajectory and sampled-event output.
                if checkpoint.stat().st_size * 3 > config["disk_gb"] * 1024**3 / task_count:
                    reason = "disk_limit"
                    break
                last_checkpoint = time.monotonic()
    except ArithmeticError as error:
        reason = "numerical_anomaly"
        atomic_json(task_dir / "error.json", {"error": repr(error)})
    except (AssertionError, ValueError) as error:
        reason = "specification_violation"
        atomic_json(task_dir / "error.json", {"error": repr(error)})
    save_checkpoint(checkpoint, state, rng, config_hash, code_hash, history, events)
    if checkpoint.stat().st_size * 3 > config["disk_gb"] * 1024**3 / task_count and reason == "normal_completion":
        reason = "disk_limit"
    final_templates = Counter(v.origin for v in state.individuals)
    summary = dict(**metadata, termination_reason=reason, completed_sweeps=state.sweep,
                   N_final=len(state.individuals), S_final=state.resource, final_by_origin=dict(final_templates),
                   extinct=not bool(state.individuals), extinction_time=state.extinction_time,
                   right_censored=state.extinction_time is None, observed_until=state.sweep,
                   elapsed_seconds=time.monotonic() - started, peak_rss_bytes=_rss_bytes(),
                   birth_events_recorded=len(events),
                   birth_event_sampling="first_events_only; aggregates include every event", **state.cumulative)
    write_csv(task_dir / "trajectory.csv", [dict(**metadata, termination_reason=reason, **row) for row in history[::max(1, int(config.get("sample_every", 1)))]])
    write_csv(task_dir / "birth_events_sample.csv", [dict(**metadata, termination_reason=reason, **row) for row in events])
    atomic_json(summary_path, summary)
    return summary


def _mean_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    radius = float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - radius, mean + radius


def run_population(output_dir, config=None, profile="smoke", resume=False, workers=1):
    """Predeclared independent replicates; never adapt to observed outcomes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = default_config(profile)
    if config:
        settings.update(config)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if len(settings["conditions"]) > 6:
        raise ValueError("At most six representative conditions")
    if settings["initial_population"] < 1 or settings["initial_population"] % 2:
        raise ValueError("Initial population must be positive and even")
    if settings["repeats"] < 1 or settings["sweeps"] < 0 or settings["S0"] < 0 or settings["S0"] > settings["Smax"]:
        raise ValueError("Invalid repetition, sweep, or initial resource count")
    if not set(settings["modes"]) <= {"single_a","single_b","mixed","pooled_shared","pooled_separate"}:
        raise ValueError("Unknown contact mode")
    if any(len(c["types"]) != 2 for c in settings["conditions"]):
        raise ValueError("Every representative condition must specify two types")
    config_hash, code_hash = stable_hash(settings), source_hash()
    run_id = f"population_{profile}_{config_hash[:12]}_{code_hash[:12]}"
    root = output_dir / run_id
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "plan.json", {"settings": settings, "config_hash": config_hash, "code_hash": code_hash,
                                     "resource_control": "pooled_shared vs pooled_separate fixes pool restrictions and matches total resources"})
    started = time.monotonic()
    deadline = started + settings["wall_seconds"]
    task_count = len(settings["conditions"]) * len(settings["modes"]) * settings["repeats"]
    tasks = [(settings, condition, mode, r, str(root), config_hash, code_hash, resume, workers, deadline, task_count)
             for condition in settings["conditions"] for mode in settings["modes"] for r in range(settings["repeats"])]
    if workers == 1:
        summaries = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(_run_task, tasks))
    write_csv(output_dir / "runs.csv", [{**row, "final_by_origin": json.dumps(row["final_by_origin"], sort_keys=True)} for row in summaries])
    aggregates = []
    for condition in settings["conditions"]:
        for mode in settings["modes"]:
            chosen = [row for row in summaries if row["condition"] == condition["name"] and row["mode"] == mode]
            for measure in ("N_final", "births", "read_failed", "resource_blocked", "deaths", "S_final"):
                mean, low, high = _mean_interval([row[measure] for row in chosen])
                aggregates.append(dict(run_id=run_id, config_hash=config_hash, stage="population_summary",
                                       condition=condition["name"], mode=mode, measure=measure,
                                       independent_runs=len(chosen), mean=mean, median=float(np.median([row[measure] for row in chosen])),
                                       mean_ci95_low=low, mean_ci95_high=high,
                                       interval_method="Student-t interval across independent run values",
                                       extinction_runs=sum(row["extinct"] for row in chosen), observation_length=settings["sweeps"],
                                       termination_reason="aggregated_all_runs"))
    write_csv(output_dir / "summary.csv", aggregates)
    contrasts = []
    for condition in settings["conditions"]:
        subset = {(row["mode"], row["repeat_id"]): row for row in summaries if row["condition"] == condition["name"]}
        for left, right in [("mixed", "single_a"), ("mixed", "single_b"), ("pooled_shared", "pooled_separate")]:
            if left not in settings["modes"] or right not in settings["modes"]:
                continue
            differences = [subset[left, r]["N_final"] - subset[right, r]["N_final"] for r in range(settings["repeats"])]
            mean, low, high = _mean_interval(differences)
            contrasts.append(dict(run_id=run_id, config_hash=config_hash, stage="population_contrast", condition=condition["name"],
                                  contrast=f"{left}-{right}", measure="N_final", independent_pairs=len(differences),
                                  mean_difference=mean, mean_ci95_low=low, mean_ci95_high=high, differences=json.dumps(differences),
                                  interval_method="paired repeat-seed differences; t interval", observation_length=settings["sweeps"],
                                  termination_reason="aggregated_all_runs"))
    write_csv(output_dir / "contrasts.csv", contrasts)
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    result = dict(run_id=run_id, config_hash=config_hash, code_hash=code_hash, profile=profile, workers=workers,
                  run_directory=str(root), task_count=len(tasks), elapsed_seconds=time.monotonic() - started, output_bytes=size,
                  all_completed=all(row["termination_reason"] == "normal_completion" for row in summaries),
                  termination_counts=dict(Counter(row["termination_reason"] for row in summaries)))
    atomic_json(output_dir / "latest.json", result)
    return result


def benchmark(output_path, trials=100000):
    """Single-process cached reads and a representative exact population pilot."""
    kernel, rng, p = Kernel(standard_chemistry(2), .5), np.random.default_rng(2026102400), (0, 0, 1, 1)
    start = time.monotonic()
    successes = sum(kernel.read_word(p, 0, p, rng) is not None for _ in range(trials))
    elapsed = time.monotonic() - start
    config = default_config("smoke")
    condition = config["conditions"][0]
    state = _initial_state(config, condition, "mixed")
    pilot_rng = np.random.default_rng(2026102401)
    exact_kernel = Kernel(_chemistry(condition["chemistry"]), condition["sigma"])
    start_pop = time.monotonic()
    for _ in range(100):
        sweep(state, exact_kernel, pilot_rng, config)
    pop_elapsed = time.monotonic() - start_pop
    value = dict(trials=trials, successes=successes, elapsed_seconds=elapsed, reads_per_second=trials / elapsed,
                 peak_rss_bytes=_rss_bytes(), workers=1, population_pilot_sweeps=100,
                 population_pilot_attempts=state.cumulative["attempted"], population_pilot_seconds=pop_elapsed,
                 estimated_default_main_sequential_seconds=pop_elapsed * 2000,
                 estimate_basis="2 conditions * 5 modes * 20 repeats * 1000 sweeps; exact pilot x2000; noise and I/O may differ",
                 output_bytes=0, rng="PCG64", seed=2026102400)
    atomic_json(output_path, value)
    value["output_bytes"] = Path(output_path).stat().st_size
    atomic_json(output_path, value)
    return value
