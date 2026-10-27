"""A deterministic same-feature counterexample to one-step compatibility.

This is an abstract word model. No biological sequence or chemical recipe is
encoded in the digital strings. Source and target order is explicit in the CSV.
"""
from collections import Counter
from pathlib import Path
import csv
import json

EXAMPLES = {
    'A0': {'g': (0,0,1,1,2,2), 'p': (0,0,1,1,2,2)},
    'A1': {'g': (2,0,0,1,1,2), 'p': (0,1,1,2,2,0)},
    'B': {'g': (0,0,1,1,0,0), 'p': (0,0,1,1,0,0)},
}

def decode(reader, template):
    lookup = {}
    for key, value in zip(reader[::2], reader[1::2]):
        if key in lookup and lookup[key] != value:
            raise ValueError('This diagnostic requires deterministic matched slots')
        lookup[key] = value
    if any(symbol not in lookup for symbol in template):
        return None
    return tuple(lookup[symbol] for symbol in template)

def reference_equal(p, q):
    return p is not None and Counter(zip(p[::2], p[1::2])) == Counter(zip(q[::2], q[1::2]))

def rows():
    for source, s in EXAMPLES.items():
        for target, t in EXAMPLES.items():
            first = decode(s['p'], t['g'])
            second = decode(first, t['g']) if first is not None else None
            yield dict(source=source, target=target, C=int(first is not None),
                       native_first=int(reference_equal(first, t['p'])),
                       S2=int(second is not None),
                       parent_copy_control_S2=int(first is not None),
                       first_child='' if first is None else ''.join(map(str,first)))

def run(root):
    out = Path(root)/'results/crossing'; out.mkdir(parents=True, exist_ok=True)
    records = list(rows())
    with (out/'same_features.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (out/'definition.json').write_text(json.dumps(dict(
        sigma=0, standard_features=True, examples=EXAMPLES,
        convention='Actor reader interprets the fixed target template; a successful child supplies the next reader. Parent-copy control retains the original actor reader.'), indent=2)+'\n')

if __name__ == '__main__':
    run(Path(__file__).resolve().parents[1])
