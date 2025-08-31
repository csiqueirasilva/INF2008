import numpy as np


def parse_label_spec(spec: str, max_id: int):
    if not spec or not spec.strip():
        return None
    keep = np.zeros(max_id + 1, dtype=bool)
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            keep[int(a):int(b) + 1] = True
        else:
            keep[int(token)] = True
    return keep

