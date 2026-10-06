"""
Cooling plant schematic drawn from a small, config-overridable component spec.

The chain follows the heat: cooling towers, cooling-tower water pumps, heat exchangers, hot-water
pumps, CDUs, racks. Pipes between stages show supply/return temperature, pressure and flow.
Everything is looked up by key in what the cooling model actually emitted, so a component the model
does not provide (the surrogate has no plant, no current FMU has chillers) is drawn greyed out as
"not modeled", and an extra component such as a chiller appears only if a spec names it and the
keys exist.
"""
import numpy as np
from rich.text import Text

# A field is (group, key, format, aggregate). group "@cdu" aggregates the per-CDU array of `key`
# with `aggregate` (mean/sum/max/min); "@sys" reads a UISnapshot attribute; other groups are
# facility groups (coolingTowerLoop, hotWaterLoop, datacenter, or a user-defined one).
DEFAULT_SPEC = {
    "components": [
        {"name": "Cooling towers", "fields": [
            ["coolingTowerLoop", "n_CTs", "{:.0f} towers", None],
            ["coolingTowerLoop", "W_flow_CT_kW", "fans {:.1f} kW", None]]},
        {"name": "CT water pumps", "fields": [
            ["coolingTowerLoop", "n_CTWPs", "{:.0f} pumps", None],
            ["coolingTowerLoop", "W_flow_CTWP_kW", "{:.1f} kW", None]]},
        {"name": "Heat exchangers", "fields": [
            ["hotWaterLoop", "n_EHXs", "{:.0f} EHX", None]]},
        {"name": "HTW pumps", "fields": [
            ["hotWaterLoop", "n_HTWPs", "{:.0f} pumps", None],
            ["hotWaterLoop", "W_flow_HTWP_kW", "{:.1f} kW", None]]},
        {"name": "CDUs", "fields": [
            ["@cdu", "W_flow_CDUP_kW", "pumps {:.1f} kW", "sum"],
            ["@cdu", "T_sec_r_C", "T ret {:.1f} C", "mean"]]},
        {"name": "Racks", "fields": [
            ["@sys", "total_power_mw", "{:.2f} MW", None],
            ["@sys", "pue", "PUE {:.3f}", None]]},
    ],
    # pipe i joins component i to component i+1: supply/return T (C), p (psig), flow (gpm).
    # Each entry is [group, key, aggregate]; `alt` is used when the group's key is missing.
    "pipes": [
        {"ts": ["coolingTowerLoop", "T_fac_ctw_s_C"], "tr": ["coolingTowerLoop", "T_fac_ctw_r_C"],
         "ps": ["coolingTowerLoop", "p_fac_ctw_s_psig"], "pr": ["coolingTowerLoop", "p_fac_ctw_r_psig"],
         "flow": ["coolingTowerLoop", "V_flow_ctw_GPM"]},
        {},  # CT water pumps -> heat exchangers share the cooling-tower loop pipe
        {},  # heat exchangers -> HTW pumps
        {"ts": ["hotWaterLoop", "T_fac_htw_s_C"], "tr": ["hotWaterLoop", "T_fac_htw_r_C"],
         "ps": ["hotWaterLoop", "p_fac_htw_s_psig"], "pr": ["hotWaterLoop", "p_fac_htw_r_psig"],
         "flow": ["hotWaterLoop", "V_flow_htw_GPM"],
         "alt": {"ts": ["@cdu", "T_prim_s_C", "mean"], "tr": ["@cdu", "T_prim_r_C", "mean"],
                 "ps": ["@cdu", "p_prim_s_psig", "mean"], "pr": ["@cdu", "p_prim_r_psig", "mean"],
                 "flow": ["@cdu", "V_flow_prim_GPM", "sum"]}},
        {"ts": ["@cdu", "T_sec_s_C", "mean"], "tr": ["@cdu", "T_sec_r_C", "mean"],
         "ps": ["@cdu", "p_sec_s_psig", "mean"], "pr": ["@cdu", "p_sec_r_psig", "mean"],
         "flow": ["@cdu", "V_flow_sec_GPM", "sum"]},
    ],
}

BOX_W = 17
PIPE_W = 15
H_WIDTH = 6 * BOX_W + 5 * PIPE_W  # width needed for the horizontal layout


def _agg(arr, how):
    f = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}[how or "mean"]
    return float(f(arr)) if np.isfinite(arr).any() else None


def _lookup(snap, group, key, how=None):
    c = snap.cooling
    if group == "@sys":
        if key == "pue":
            return c.pue if c is not None else None
        return getattr(snap, key, None)
    if c is None:
        return None
    if group == "@cdu":
        arr = c.cdu.get(key)
        return _agg(arr, how) if arr is not None else None
    return c.facility.get(group, {}).get(key)


def _pipe_values(snap, pipe):
    vals = {k: _lookup(snap, *pipe[k]) for k in ("ts", "tr", "ps", "pr", "flow") if k in pipe}
    if pipe.get("alt") and all(v is None for v in vals.values()):
        vals = {k: _lookup(snap, *pipe["alt"][k]) for k in pipe["alt"]}
        vals["approx"] = True
    return vals


def _fmt(v, f):
    return f.format(v) if v is not None else "n/a"


def _component_lines(snap, comp):
    """(lines, modeled): modeled is False if none of the component's keys are present."""
    lines, got = [], False
    for group, key, fmt, how in comp["fields"]:
        v = _lookup(snap, group, key, how)
        got = got or v is not None
        lines.append(_fmt(v, fmt))
    return lines, got


def render_schematic(snap, width, spec=None):
    """Rich Text of the plant for the cooling snapshot `snap.cooling`, laid out for `width` columns."""
    spec = spec or DEFAULT_SPEC
    comps, pipes = spec["components"], spec["pipes"]
    pipe_vals = [_pipe_values(snap, p) if p else None for p in pipes]
    # Pipes without their own entry inherit the previous one (same loop)
    for i, pv in enumerate(pipe_vals):
        if pv is None and i > 0:
            pipe_vals[i] = pipe_vals[i - 1]
    parts = [_component_lines(snap, c) for c in comps]
    if width >= len(comps) * BOX_W + (len(comps) - 1) * PIPE_W:
        return _horizontal(comps, parts, pipe_vals)
    return _vertical(comps, parts, pipe_vals)


def _pipe_label(v):
    if v is None or all(v.get(k) is None for k in ("ts", "tr", "ps", "pr", "flow")):
        return None
    pt = f"{_fmt(v.get('ts'), '{:.1f}')}/{_fmt(v.get('tr'), '{:.1f}')} C"
    pp = f"{_fmt(v.get('ps'), '{:.0f}')}/{_fmt(v.get('pr'), '{:.0f}')} psi"
    fl = f"{_fmt(v.get('flow'), '{:,.0f}')} gpm"
    return pt, pp, fl, v.get("approx", False)


def _horizontal(comps, parts, pipe_vals):
    rows = 6
    t = Text()
    for r in range(rows):
        for i, comp in enumerate(comps):
            lines, modeled = parts[i]
            style = "cyan" if modeled else "dim"
            if r == 0:
                t.append("┌" + "─" * (BOX_W - 2) + "┐", style=style)
            elif r == rows - 1:
                t.append("└" + "─" * (BOX_W - 2) + "┘", style=style)
            else:
                if r == 1:
                    txt, tstyle = comp["name"], "bold" if modeled else "dim"
                else:
                    k = r - 2
                    txt = (lines[k] if modeled else ("not modeled" if k == 0 else "")) if k < max(len(lines), 1) else ""
                    tstyle = "" if modeled else "dim italic"
                t.append("│", style=style)
                t.append(txt[:BOX_W - 2].center(BOX_W - 2), style=tstyle)
                t.append("│", style=style)
            if i < len(comps) - 1:
                t.append(_pipe_cell(pipe_vals[i], r, rows))
        t.append("\n")
    return t


def _pipe_cell(v, r, rows):
    lab = _pipe_label(v)
    w = PIPE_W
    if r == 1:
        return Text((lab[0] if lab else "").center(w), style="dim" if lab is None else "")
    if r == 2:
        return Text("─" * (w - 1) + "►", style="blue" if lab else "dim")
    if r == 3:
        return Text("◄" + "─" * (w - 1), style="red" if lab else "dim")
    if r == 4:
        return Text(((lab[1] if lab else "")[:w]).center(w), style="dim")
    if r == 5:
        return Text(((lab[2] if lab else "")[:w]).center(w), style="dim")
    return Text(" " * w)


def _vertical(comps, parts, pipe_vals):
    t = Text()
    for i, comp in enumerate(comps):
        lines, modeled = parts[i]
        t.append("■ ", style="cyan" if modeled else "dim")
        t.append(f"{comp['name']:<16}", style="bold" if modeled else "dim")
        t.append("  ".join(lines) if modeled else "not modeled", style="" if modeled else "dim italic")
        t.append("\n")
        if i < len(comps) - 1:
            lab = _pipe_label(pipe_vals[i])
            t.append("   ↓ ", style="blue")
            if lab:
                t.append(f"S/R {lab[0]}  {lab[1]}  {lab[2]}" + (" (CDU side)" if lab[3] else ""), style="dim")
            t.append(" ↑", style="red")
            t.append("\n")
    return t
