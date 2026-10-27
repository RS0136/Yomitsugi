"""Independently replay the two residual-symmetry witnesses using literal slots.

This verifier does not import the group-dynamics or diagnostic implementation.
All relabelings are enumerated for the two fixed targets; no additional target
family or population experiment is introduced.
"""
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def renamed(code, labels):
    table = {labels[key]: labels[value] for key, value in enumerate(code)}
    return tuple(table[key] for key in range(len(code)))


def literal_child(code, description):
    output = [code[symbol] for symbol in description]
    table = dict(zip(output[::2], output[1::2]))
    assert len(table) == len(code)
    return tuple(table[key] for key in range(len(code)))


def main():
    source = ROOT / 'results/relabeling_diagnostic/orbits.json'
    saved = json.loads(source.read_text())
    cases = {}
    for name, expected_period, expected_quotient in [('q3', 2, 1), ('q8', 192, 16)]:
        record = saved[name]
        target, initial = tuple(record['target']), tuple(record['initial'])
        size = len(target)
        centralizer = [c for c in permutations(range(size))
                       if all(c[target[i]] == target[c[i]] for i in range(size))]
        assert set(centralizer) == {tuple(c) for c in record['centralizer']}
        description = [symbol for k in range(size) for symbol in (k, target[k])]
        code, cycle = initial, []
        while code not in cycle:
            cycle.append(code)
            code = literal_child(code, description)
        assert code == initial
        assert cycle == [tuple(p) for p in record['cycle']]
        assert len(cycle) == expected_period
        projected = [min(renamed(p, c) for c in centralizer) for p in cycle]
        native_class = min(renamed(target, c) for c in centralizer)
        assert native_class == target and native_class not in projected
        quotient = next(h for h in range(1, expected_period + 1)
                        if all(projected[j] == projected[(j+h) % expected_period]
                               for j in range(expected_period)))
        assert quotient == expected_quotient == record['quotient_period']
        assert len(set(projected)) == quotient
        for c in centralizer:
            assert [renamed(p, c) for p in cycle] == [
                literal_child(renamed(cycle[(j-1) % expected_period], c), description)
                for j in range(expected_period)]
        shifts = [c for c in centralizer if renamed(initial, c) == cycle[quotient % expected_period]]
        assert set(shifts) == {tuple(c) for c in record['quotient_shift_witnesses']}
        cases[name] = {'labelled_period': len(cycle), 'quotient_period': quotient,
                       'centralizer_size': len(centralizer),
                       'native_class_is_singleton': True,
                       'native_equivalent_to_cycle': False,
                       'all_saved_states_and_shift_witnesses_match': True}
    result = {'method': 'Independent literal-slot replay and exhaustive relabelings of two fixed targets',
              'cases': cases, 'all_passed': True,
              'source_sha256': sha256(source.read_bytes()).hexdigest(),
              'diagnostic_source_sha256': sha256((ROOT/'analysis/relabeling_diagnostic.py').read_bytes()).hexdigest(),
              'verifier_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = ROOT/'results/validation/relabeling_verification.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
