#load file info:
import json

def numeric_diff(a: str, b: str):
    i = 0
    diffs = []

    while i < len(a):
        if a[i] == b[i]:
            i += 1
            continue

        if not (a[i].isdigit() and b[i].isdigit()):
            return None

        start = i
        while start > 0 and a[start - 1].isdigit() and b[start - 1].isdigit():
            start -= 1

        end = i
        while end < len(a) and a[end].isdigit() and b[end].isdigit():
            end += 1

        diffs.append((start, end, int(a[start:end]), int(b[start:end])))
        i = end

    if not diffs:
        return None

    # Try each differing number as dynamic base
    for  (_, _, base_a, base_b) in diffs:

        # -------------------------------------------------
        # 1) PURE / MULTIPLICATIVE: value = dynamic * k
        # -------------------------------------------------
        mult_valid = True
        mult_factors = []

        for _, _, x, y in diffs:
            if x == base_a and y == base_b:
                mult_factors.append(1)
                continue

            if x % base_a != 0 or y % base_b != 0:
                mult_valid = False
                break

            kx = x // base_a
            ky = y // base_b

            if kx != ky:
                mult_valid = False
                break

            mult_factors.append(kx)

        if mult_valid:

            normalized = []
            last = 0
            for (start, end, _, _), factor in zip(diffs, mult_factors):
                normalized.append(a[last:start])
                if factor == 1:
                    normalized.append("XXX")
                else:
                    normalized.append(f"XXX*{factor}")
                last = end
            normalized.append(a[last:])

            return "".join(normalized), base_a, base_b

        # -------------------------------------------------
        # 2) ADDITIVE: value = dynamic + C
        # -------------------------------------------------
        add_valid = True
        deltas = []

        for _, _, x, y in diffs:
            dx = x - base_a
            dy = y - base_b
            if dx != dy:
                add_valid = False
                break
            deltas.append(dx)

        if add_valid:
            normalized = []
            last = 0
            for (start, end, _, _), delta in zip(diffs, deltas):
                normalized.append(a[last:start])
                if delta == 0:
                    normalized.append("XXX")
                else:
                    normalized.append(f"XXX{delta:+d}")
                last = end
            normalized.append(a[last:])

            return "".join(normalized), base_a, base_b

    return None





from collections import defaultdict

def build_groups(operations):
    groups = defaultdict(dict)  # key → {dynamic_value: op}

    for i, op_a in enumerate(operations):
        for j in range(i + 1, len(operations)):
            op_b = operations[j]

            res = numeric_diff(op_a["label"], op_b["label"])
            if not res:
                continue

            key, val_a, val_b = res

            groups[key][val_a] = op_a
            groups[key][val_b] = op_b

    return groups

from collections import Counter

def infer_step(values):
    if len(values) < 2:
        return None

    diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not diffs:
        return None

    # Most common positive difference
    step, _ = Counter(diffs).most_common(1)[0]
    return step

def split_on_structure(group_dict, max_hole=1, min_points=3):
    """
    Input: {dynamic_value: op}
    Output: list of structured groups:
      {
        "values": {dynamic_value: op},
        "step": inferred step,
        "holes": hole_count
      }
    """
    values = sorted(group_dict.keys())
    result = []

    i = 0
    while i < len(values):
        current_vals = [values[i]]
        j = i + 1

        while j < len(values):
            trial = current_vals + [values[j]]
            step = infer_step(trial)

            if step is None:
                break

            expected = list(range(trial[0], trial[-1] + 1, step))
            holes = len(expected) - len(trial)

            if holes > max_hole:
                break

            current_vals.append(values[j])
            j += 1

        step = infer_step(current_vals)
        if step is not None:
            expected = list(range(current_vals[0], current_vals[-1] + 1, step))
            holes = len(expected) - len(current_vals)
        else:
            holes = 0

        # 🔴 NEW: require enough points to justify a step
        if step is not None and len(current_vals) < min_points:
            # Treat each value as its own singleton
            for v in current_vals:
                result.append({
                    "values": {v: group_dict[v]},
                    "step": None,
                    "holes": 0
                })
        else:
            result.append({
                "values": {v: group_dict[v] for v in current_vals},
                "step": step,
                "holes": holes
            })
        #expected = list(range(current_vals[0], current_vals[-1] + 1, step))
        #holes = len(expected) - len(current_vals)

        # result.append({
        #     "values": {v: group_dict[v] for v in current_vals},
        #     "step": step,
        #     "holes": holes
        # })

        i = j

    return result

def split_on_holes(group_dict):
    """
    Input: {dynamic_value: op}
    Output: list of {dynamic_value: op} groups
    """
    values = sorted(group_dict.keys())
    result = []

    current = {values[0]: group_dict[values[0]]}

    for prev, curr in zip(values, values[1:]):
        if curr == prev + 1:
            current[curr] = group_dict[curr]
        else:
            result.append(current)
            current = {curr: group_dict[curr]}

    result.append(current)
    return result

def finalize_groups(candidate_groups, all_ops):
    final_groups = []
    singles = {id(op): op for op in all_ops}

    for key, group_dict in candidate_groups.items():
        split_groups = split_on_structure(group_dict)

        for g in split_groups:
            if len(g["values"]) > 1:
                final_groups.append((key, g))
                for op in g["values"].values():
                    singles.pop(id(op), None)

    return final_groups, list(singles.values())

from statistics import mean

def group_stats(group):
    dynamics = sorted(group["values"].keys())
    values = [
        group["values"][d]["count"] * group["values"][d]["mean"]
        for d in dynamics
    ]

    return {
        "min": dynamics[0],
        "max": dynamics[-1],
        "step": group["step"],
        "holes": group["holes"],
        "mean": mean(values),
        "total": sum(values),
        "steepness": (
            (values[-1] - values[0]) / (dynamics[-1] - dynamics[0])
            if dynamics[-1] != dynamics[0] else 0.0
        )
    }

def analyse_file(data):
    # Separate the data by measurement type
    duration_data = [op for op in data if op['m_type'] == 'Duration']
    output_size_data = [op for op in data if op['m_type'] == 'OutputSize']
    dispatch_size_data = [op for op in data if op['m_type'] == 'DispatchSize']

    # Function to process data for plotting
    def process_data(data):
        # Precompute values
        for op in data:
            op['value'] = op['count'] * op['mean']

        # ---- NEW: grouping pipeline ----
        candidate_groups = build_groups(data)
        groups, singles = finalize_groups(candidate_groups, data)

        aggregated = []

        # Add grouped entries
        for key, group in groups:
            stats = group_stats(group)

            aggregated.append({
                "label": key,
                "value": stats["total"],
                "count": sum(op["count"] for op in group["values"].values()),
                "group": True,
                "stats": stats
            })

        # Add ungrouped singles (only once)
        for op in singles:
            aggregated.append({
                "label": op["label"],
                "value": op["value"],
                "count": op["count"],
                "group": False
            })

        # ---- Sort like before ----
        aggregated.sort(key=lambda x: x["value"], reverse=True)

        total_sum = sum(a["value"] for a in aggregated)

        print(f"Total sum for {data[0]['m_type']}: {total_sum}")
        print(f"Operations sorted by total {data[0]['m_type'].lower()}:")

        for entry in aggregated:
            perc = entry["value"] / total_sum if total_sum else 0

            if entry["group"]:
                s = entry["stats"]
                print(
                    f"[GROUP] {entry['label']}\n"
                    f"  Range: {s['min']} → {s['max']}(step={s['step']}, holes={s['holes']})\n"
                    f"  Total: {entry['value']:.6f} | "
                    f"Mean: {s['mean']:.6f} | "
                    f"Steepness: {s['steepness']:.3e}\n"
                    f"  Count: {entry['count']} | Perc: {perc:.2%}"
                )
            else:
                print(
                    f"Operation: {entry['label']}, "
                    f"{data[0]['m_type']}: {entry['value']:.6f}, "
                    f"Count: {entry['count']} "
                    f"Perc:{perc:.2%}"
                )

        # Labels/values still returned for plotting
        labels = [e["label"] for e in aggregated]
        values = [e["value"] for e in aggregated]

        return labels, values

    # Process data for each measurement type
    duration_labels, duration_values = process_data(duration_data)
    # output_size_labels, output_size_values = process_data(output_size_data)
    # dispatch_size_labels, dispatch_size_values = process_data(dispatch_size_data)


# Load JSON data from a file
with open('path_to_wgpu_debug_file_measurements.json', 'r') as file:
    data = json.load(file)

analyse_file(data)


