"""Frozen pre-optimization sweep, retained for exact pathwise validation."""
from collections import Counter
import json
import numpy as np
from .population import Individual, _split_integer

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
                        "b": child.b, "g": json.dumps(child.g), "p": json.dumps(child.p)})
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
    template_counts = Counter(f"{v.b}:{','.join(map(str, v.g))}" for v in state.individuals)
    origin_counts = Counter(v.origin for v in state.individuals)
    reader_counts = Counter(",".join(map(str, v.p)) for v in state.individuals)
    joint_counts = Counter(f"{v.b}:{','.join(map(str,v.g))}|{','.join(map(str,v.p))}" for v in state.individuals)
    return dict(sweep=state.sweep, N_before=n, N=len(state.individuals), S_before=s_before,
                S=state.resource, offered=J, **increments,
                origin_counts=json.dumps(dict(origin_counts), sort_keys=True),
                template_counts=json.dumps(dict(template_counts), sort_keys=True),
                reader_counts=json.dumps(dict(reader_counts), sort_keys=True),
                template_reader_counts=json.dumps(dict(joint_counts), sort_keys=True),
                direction_births=json.dumps(dict(directional), sort_keys=True),
                distinct_readers=len(reader_counts), extinct=not bool(state.individuals))



