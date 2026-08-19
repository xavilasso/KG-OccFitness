#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constructor del Fitness Knowledge Graph, version 3.

  --source free-exercise-db   (RECOMENDADA) ingiere el JSON de
      https://github.com/yuhonas/free-exercise-db  (dominio publico, Unlicense),
      que trae primaryMuscles, secondaryMuscles, equipment, force, mechanic,
      level, category e instructions por ejercicio. Los atributos vienen del
      dataset, no de heuristicas sobre el nombre.

  --source names              modo compatible: lista plana de nombres +
      heuristicas por palabra clave (lo que hacia la v2). Se conserva solo para
      reproducir el grafo anterior; NO deberia usarse para publicar.

Uso:
    python build_fitkg_v3.py --source free-exercise-db --input exercises.json
    python build_fitkg_v3.py --source names --input exercises_full_v2.json --allow-warnings
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
    PYG = True
except Exception:
    PYG = False


# =====================================================================
# Esquema: dominio y rango de cada relacion. Se valida al final.
# =====================================================================

SCHEMA = {
    "targets":       ("Exercise", "MuscleGroup"),
    "recruits":      ("Exercise", "MuscleGroup"),
    "requires":      ("Exercise", "Equipment"),
    "hasType":       ("Exercise", "ExerciseType"),
    "hasIntensity":  ("Exercise", "IntensityLevel"),
    "supportsGoal":  ("Exercise", "ExerciseGoal"),
    "isVariationOf": ("Exercise", "Exercise"),
    # Dos relaciones nuevas que salen DIRECTO del dataset, sin heuristicas:
    "hasForce":      ("Exercise", "ForceType"),      # push / pull / static
    "hasMechanic":   ("Exercise", "MechanicType"),   # compound / isolation
}

# Calificadores de equipamiento al inicio del nombre. Se retiran para detectar la familia de movimiento a la que pertenece un ejercicio.
EQUIPMENT_QUALIFIER = re.compile(
    r"^(barbell|dumbbell|cable|machine|smith machine|kettlebell|band|bands|banded|"
    r"bodyweight|body weight|ez.?bar|e-z.?bar|lever|sled|weighted|seated|standing|"
    r"lying|incline|decline|reverse|single.?arm|one.?arm|alternate|alternating)\s+",
    re.I)

VARIATION_SUFFIXES = [
    "Incline", "Decline", "Single Arm", "Double KB", "Weighted", "Bodyweight",
    "Assisted", "Pause Rep", "Slow Eccentric", "Tempo 3-1-3",
]


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).lower()


class Graph:
    """Namespace por (tipo, nombre normalizado). E1."""

    def __init__(self):
        self.nodes = []
        self._index = {}
        self.edges = []
        self._edge_seen = set()
        self.warnings = []

    def add_node(self, name, ntype, attrs=None):
        key = (ntype, norm(name))
        if key in self._index:
            return self._index[key]
        nid = len(self.nodes)
        self._index[key] = nid
        self.nodes.append({"node_id": nid, "name": name, "type": ntype,
                           "attrs": attrs or {}})
        return nid

    def get(self, name, ntype):
        return self._index.get((ntype, norm(name)))

    def add_edge(self, src, dst, rel):
        if src is None or dst is None:
            self.warnings.append(f"arista {rel} descartada: extremo inexistente")
            return False
        if src == dst:                                              # E4
            self.warnings.append(
                f"self-loop descartado: {self.nodes[src]['name']} --{rel}--> si mismo")
            return False
        key = (src, dst, rel)
        if key in self._edge_seen:
            return False
        self._edge_seen.add(key)
        self.edges.append({"source": src, "target": dst, "rel": rel})
        return True


# =====================================================================
# Fuente 1 (recomendada): free-exercise-db
# =====================================================================

LEVEL_TO_INTENSITY = {"beginner": "Low", "intermediate": "Moderate", "expert": "High"}

CATEGORY_TO_TYPE = {
    "strength": "Strength", "stretching": "Flexibility", "plyometrics": "Plyometrics",
    "strongman": "Power", "powerlifting": "Power", "cardio": "Cardio",
    "olympic weightlifting": "Power",
}

CATEGORY_TO_GOALS = {
    "strength": ["Strength", "Hypertrophy"], "powerlifting": ["Strength"],
    "olympic weightlifting": ["Power", "Strength"], "strongman": ["Power", "Strength"],
    "cardio": ["Endurance", "Fat Loss"], "stretching": ["Mobility", "Rehab"],
    "plyometrics": ["Power"],
}


def load_free_exercise_db(path, curated_muscles=None):
    """Devuelve registros normalizados desde el JSON de free-exercise-db."""
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("exercises", list(raw.values()))

    recs = []
    for e in raw:
        name = e.get("name")
        if not name:
            continue
        cat = norm(e.get("category") or "strength")
        primary = [m.title() for m in (e.get("primaryMuscles") or [])]
        if curated_muscles and name in curated_muscles:      # E6: curacion manda
            primary = curated_muscles[name]
        recs.append(dict(
            name=name,
            primary=primary,
            secondary=[m.title() for m in (e.get("secondaryMuscles") or [])],
            equipment=(e.get("equipment") or "body only").title(),
            etype=CATEGORY_TO_TYPE.get(cat, "Strength"),
            intensity=LEVEL_TO_INTENSITY.get(norm(e.get("level") or ""), "Moderate"),
            goals=CATEGORY_TO_GOALS.get(cat, ["Strength"]),
            source_id=e.get("id"),
            force=e.get("force"), mechanic=e.get("mechanic"), level=e.get("level"),
            curated=bool(curated_muscles and name in curated_muscles),
        ))
    return recs


# =====================================================================
# Fuente 2 (compatibilidad): lista de nombres + heuristicas de la v2
# =====================================================================

def load_name_list(path, heuristics_module, curated_muscles=None):
    names = json.load(open(path, encoding="utf-8"))
    if isinstance(names, dict):
        names = names.get("exercises", [])
    h = heuristics_module
    recs = []
    for n in names:
        primary = curated_muscles.get(n) if curated_muscles else None
        recs.append(dict(
            name=n,
            primary=primary or h.infer_primary_muscles(n),
            secondary=[],
            equipment=h.infer_equipment(n),
            etype=h.infer_type(n),
            intensity=h.infer_intensity(n),
            goals=h.infer_goals(n),
            source_id=None, force=None, mechanic=None, level=None,
            curated=bool(primary),
        ))
    return recs


# =====================================================================
# Construccion
# =====================================================================

def strip_variation_suffix(name):
    """'Barbell Squat - Pause Rep' -> ('Barbell Squat', 'Pause Rep'). E2."""
    for suf in VARIATION_SUFFIXES:
        for sep in (" - ", " – ", ": "):
            if name.endswith(sep + suf):
                return name[: -len(sep + suf)], suf
    for pre in ("Incline ", "Decline "):
        if name.startswith(pre):
            return name[len(pre):], pre.strip()
    return None, None


def movement_core(name):
    """'Barbell Bench Press - Medium Grip' -> 'bench press'.

    Retira el sufijo de variante y los calificadores de equipamiento o postura
    del inicio, para agrupar el mismo patron de movimiento ejecutado con
    distintos implementos.
    """
    base, _ = strip_variation_suffix(name)
    core = base or name
    core = core.split(" - ")[0]
    prev = None
    while prev != core:
        prev = core
        core = EQUIPMENT_QUALIFIER.sub("", core).strip()
    return norm(core)


def build(recs, add_goals=True, variation_mode="family"):
    g = Graph()
    primary_by_id = {}

    for r in recs:
        eid = g.add_node(r["name"], "Exercise", {
            k: r[k] for k in ("source_id", "force", "mechanic", "level", "curated")})
        primary_by_id[eid] = {norm(m) for m in r["primary"]}

        for m in r["primary"]:
            g.add_edge(eid, g.add_node(m, "MuscleGroup"), "targets")
        for m in r["secondary"]:
            if m not in r["primary"]:
                g.add_edge(eid, g.add_node(m, "MuscleGroup"), "recruits")
        g.add_edge(eid, g.add_node(r["equipment"], "Equipment"), "requires")
        g.add_edge(eid, g.add_node(r["etype"], "ExerciseType"), "hasType")
        g.add_edge(eid, g.add_node(r["intensity"], "IntensityLevel"), "hasIntensity")
        if r.get("force"):
            g.add_edge(eid, g.add_node(str(r["force"]).title(), "ForceType"), "hasForce")
        if r.get("mechanic"):
            g.add_edge(eid, g.add_node(str(r["mechanic"]).title(), "MechanicType"), "hasMechanic")
        if add_goals:
            for go in r["goals"]:
                g.add_edge(eid, g.add_node(go, "ExerciseGoal"), "supportsGoal")

    # --- isVariationOf ------------------------------------------------
    for n in list(g.nodes):
        if n["type"] != "Exercise":
            continue
        base, _ = strip_variation_suffix(n["name"])
        if base:
            bid = g.get(base, "Exercise")
            if bid is not None:
                g.add_edge(n["node_id"], bid, "isVariationOf")

    # La base canonica es el miembro de nombre mas corto. Estas aristas son CANDIDATAS y se exportan aparte para validacion por el experto.
    candidates = []
    if variation_mode == "family":
        fams = defaultdict(list)
        for n in g.nodes:
            if n["type"] == "Exercise":
                fams[movement_core(n["name"])].append(n["node_id"])
        for core, members in fams.items():
            if len(members) < 2 or not core:
                continue
            base = min(members, key=lambda i: (len(g.nodes[i]["name"]), g.nodes[i]["name"]))
            for m in members:
                if m == base:
                    continue
                if primary_by_id[m] & primary_by_id[base]:
                    if g.add_edge(m, base, "isVariationOf"):
                        candidates.append({
                            "variant": g.nodes[m]["name"], "base": g.nodes[base]["name"],
                            "movement_core": core, "expert_verdict": ""})
    g.variation_candidates = candidates
    return g


# =====================================================================
# Validacion
# =====================================================================

def validate(g, strict=True):
    errors, notes = [], []
    ntype = {n["node_id"]: n["type"] for n in g.nodes}
    nname = {n["node_id"]: n["name"] for n in g.nodes}

    for e in g.edges:
        dom, rng = SCHEMA[e["rel"]]
        if ntype[e["source"]] != dom:
            errors.append(f"{e['rel']}: origen '{nname[e['source']]}' es "
                          f"{ntype[e['source']]}, se esperaba {dom}")
        if ntype[e["target"]] != rng:
            errors.append(f"{e['rel']}: destino '{nname[e['target']]}' es "
                          f"{ntype[e['target']]}, se esperaba {rng}")

    deg = Counter()
    for e in g.edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    isolated = [(nname[n["node_id"]], n["type"]) for n in g.nodes if deg[n["node_id"]] == 0]
    if isolated:
        notes.append(f"{len(isolated)} nodos aislados: {isolated}")

    by_rel = defaultdict(lambda: {"h": set(), "t": set()})
    for e in g.edges:
        by_rel[e["rel"]]["h"].add(e["source"])
        by_rel[e["rel"]]["t"].add(e["target"])
    for rel, s in sorted(by_rel.items()):
        dom, rng = SCHEMA[rel]
        total_rng = sum(1 for n in g.nodes if n["type"] == rng)
        notes.append(f"{rel:14s} aristas={sum(1 for e in g.edges if e['rel'] == rel):5d} "
                     f"dominio={len(s['h']):4d} rango={len(s['t']):3d}/{total_rng:3d}")

    curated = sum(1 for n in g.nodes if n["type"] == "Exercise" and n["attrs"].get("curated"))
    n_ex = sum(1 for n in g.nodes if n["type"] == "Exercise")
    notes.append(f"ejercicios con curacion manual explicita: {curated}/{n_ex}")

    print("\n=== VALIDACION ===")
    for x in notes:
        print("  ·", x)
    for w in g.warnings[:20]:
        print("  ! ", w)
    if len(g.warnings) > 20:
        print(f"  ! ... y {len(g.warnings) - 20} avisos mas")
    if errors:
        print(f"\n  ERRORES DE ESQUEMA: {len(errors)}")
        for x in errors[:15]:
            print("   x", x)
        if strict:
            sys.exit("\nAbortado: el grafo viola el esquema. Corrige la fuente o "
                     "vuelve a correr con --allow-warnings.")
    else:
        print("\n  esquema: OK, ninguna relacion fuera de su dominio/rango")
    return errors, isolated


# =====================================================================
# Exportacion
# =====================================================================

def export(g, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nodes_df = pd.DataFrame([{**n, "attrs_json": json.dumps(n["attrs"], ensure_ascii=False)}
                             for n in g.nodes])[["node_id", "name", "type", "attrs_json"]]
    nodes_df.to_csv(os.path.join(out_dir, "nodes.csv"), index=False, encoding="utf-8-sig")
    edges_df = pd.DataFrame(g.edges)
    edges_df.to_csv(os.path.join(out_dir, "edges.csv"), index=False, encoding="utf-8-sig")

    cands = getattr(g, "variation_candidates", [])
    if cands:
        pd.DataFrame(cands).to_csv(
            os.path.join(out_dir, "isvariationof_para_validar.csv"),
            index=False, encoding="utf-8-sig")
        print(f"\n{len(cands)} aristas isVariationOf candidatas escritas en "
              f"isvariationof_para_validar.csv (columna expert_verdict vacia)")

    summary = {
        "num_nodes": len(g.nodes),
        "num_assertions": len(g.edges),
        "node_types": dict(Counter(n["type"] for n in g.nodes)),
        "relation_types": dict(Counter(e["rel"] for e in g.edges)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not PYG:
        print("\nPyG no disponible: no se escribe graph_data.pt")
        return

    node_types = sorted({n["type"] for n in g.nodes})
    t2i = {t: i for i, t in enumerate(node_types)}
    X = []
    for n in g.nodes:
        oh = [0.0] * len(node_types)
        oh[t2i[n["type"]]] = 1.0
        X.append(oh)
    x = torch.tensor(X, dtype=torch.float)

    rels = sorted({e["rel"] for e in g.edges})
    r2i = {r: i for i, r in enumerate(rels)}

    edge_index = torch.tensor([[e["source"] for e in g.edges],
                               [e["target"] for e in g.edges]], dtype=torch.long)
    edge_type = torch.tensor([r2i[e["rel"]] for e in g.edges], dtype=torch.long)

    torch.save(Data(x=x, edge_index=edge_index, edge_type=edge_type),
               os.path.join(out_dir, "graph_data.pt"))
    for fname, obj in (("edge_rel_mapping.json", r2i), ("node_type_mapping.json", t2i)):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"\ngraph_data.pt escrito con {edge_index.size(1)} aristas dirigidas "
          f"(sin espejo) y {len(rels)} relaciones")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["free-exercise-db", "names"],
                   default="free-exercise-db")
    p.add_argument("--input", required=True)
    p.add_argument("--curated", help="JSON opcional {nombre: [musculos primarios]} "
                                     "con la curacion del experto")
    p.add_argument("--heuristics", default="build_fitkg_lite_v2",
                   help="modulo con infer_* para --source names")
    p.add_argument("--out", default="fitkg_v3_output")
    p.add_argument("--no-goals", dest="goals", action="store_false")
    p.add_argument("--allow-warnings", dest="strict", action="store_false")
    a = p.parse_args()

    curated = json.load(open(a.curated, encoding="utf-8")) if a.curated else None
    if curated:
        print(f"curacion manual cargada: {len(curated)} ejercicios")

    if a.source == "free-exercise-db":
        recs = load_free_exercise_db(a.input, curated)
    else:
        import importlib
        recs = load_name_list(a.input, importlib.import_module(a.heuristics), curated)
    print(f"registros cargados: {len(recs)}")

    g = build(recs, add_goals=a.goals)
    validate(g, strict=a.strict)
    export(g, a.out)
    print("\nListo.")


if __name__ == "__main__":
    main()