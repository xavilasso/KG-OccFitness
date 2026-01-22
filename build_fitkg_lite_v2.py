import os
import json, re
import pandas as pd
from tqdm import tqdm
import torch
import networkx as nx
from collections import defaultdict

# Try import PyG; if not available, will still produce CSVs
try:
    from torch_geometric.data import Data
    import torch_geometric
    pyg_available = True
except Exception as e:
    print("torch_geometric not available; will still output CSVs. Install PyG to generate .pt graph.")
    pyg_available = False

output_dir = "fitkg_lite_output"
os.makedirs(output_dir, exist_ok=True)

# VOCAB: Muscles, Equip, Types, Intensity, Goals
muscle_groups = [
    "Quadriceps","Hamstrings","Glutes","Calves","Hip Flexors",
    "Chest","Upper Back","Lats","Lower Back",
    "Deltoids Anterior","Deltoids Lateral","Deltoids Posterior","Deltoids",
    "Biceps","Triceps","Forearms",
    "Core","Obliques","Lower Abs","Upper Abs",
    "Adductors","Abductors","Serratus Anterior","Neck"
]

equipment_list = [
    "Barbell","Dumbbell","Kettlebell","Cable Machine","Resistance Band",
    "Bodyweight","Bench","Smith Machine","Leg Press Machine","Pull-up Bar",
    "Stability Ball","Medicine Ball","TRX","Rowing Machine","Treadmill","None"
]

exercise_types = [
    "Strength","Hypertrophy","Power","Endurance","Cardio","Mobility","Flexibility","Stability","Balance","Plyometrics"
]

intensities = ["Low","Moderate","High"]

exercise_goals = ["Strength","Hypertrophy","Fat Loss","Endurance","Mobility","Rehab","Power"]

# EXERCISES LIST (380 entries)
# categories (legs, chest, back, shoulders, arms, core, cardio, mobility)

JSON_PATH = "exercises_full_v2.json"

with open(JSON_PATH, "r") as f:
    exercises = json.load(f)["exercises"]

print(f"Loaded {len(exercises)} exercises from JSON.")


# Mapping heuristics (some primary muscle mapping examples)
# For simplicity, we'll define a mapping for many of the common exercises.
# For others the script will apply heuristics (keyword matching).

primary_muscle_map = {
    # Legs examples
    "Barbell Back Squat": ["Quadriceps","Glutes","Hamstrings"],
    "Front Squat": ["Quadriceps","Glutes"],
    "Goblet Squat": ["Quadriceps","Glutes"],
    "Bulgarian Split Squat": ["Quadriceps","Glutes"],
    "Walking Lunge": ["Quadriceps","Glutes","Hamstrings"],
    "Romanian Deadlift": ["Hamstrings","Glutes"],
    "Deadlift": ["Hamstrings","Glutes","Lower Back"],
    "Hip Thrust": ["Glutes"],
    "Glute Bridge": ["Glutes","Hamstrings"],
    "Leg Press": ["Quadriceps","Glutes"],
    "Calf Raise Standing": ["Calves"],
    "Pistol Squat": ["Quadriceps","Glutes"],
    # Chest
    "Barbell Bench Press": ["Chest","Triceps","Deltoids Anterior"],
    "Dumbbell Bench Press": ["Chest","Triceps"],
    "Push Up": ["Chest","Triceps","Deltoids Anterior"],
    "Incline Bench Press": ["Chest","Deltoids Anterior"],
    # Back
    "Pull-Up": ["Lats","Upper Back","Biceps"],
    "Lat Pulldown": ["Lats"],
    "Seated Cable Row": ["Upper Back","Lats"],
    "Bent Over Barbell Row": ["Upper Back","Lats"],
    # Shoulders
    "Overhead Press": ["Deltoids Anterior","Deltoids Lateral","Triceps"],
    "Lateral Raise": ["Deltoids Lateral"],
    "Rear Delt Fly": ["Deltoids Posterior"],
    # Arms
    "Barbell Curl": ["Biceps"],
    "Triceps Pushdown": ["Triceps"],
    # Core
    "Plank": ["Core"],
    "Russian Twist": ["Obliques"],
    "Dead Bug": ["Core"],
    # Cardio
    "Running (Treadmill)": ["Quadriceps","Calves"],
    "Cycling (Stationary)": ["Quadriceps","Calves"],
    # Power
    "Hang Clean": ["Hamstrings","Glutes","Upper Back"],
    "Power Clean": ["Hamstrings","Glutes","Upper Back"],
    # Misc
    "Turkish Get Up": ["Core","Shoulders","Glutes"]
}

def normalize_name(name):
    return re.sub(r"\s+", " ", name.lower().strip())

# utility heuristics

# Change to semantic family relation to elevate the quality of the classification
def infer_semantic_family(name: str) -> str:
    n = name.lower()

    # Weightlifting / Olympic
    if any(k in n for k in [
        "snatch", "clean", "jerk", "hang clean", "hang snatch", "power clean", "power snatch"
    ]):
        return "weightlifting"

    # Carries
    if any(k in n for k in [
        "carry", "farmer", "suitcase", "overhead carry"
    ]):
        return "carry"

    # Core rotational / anti-rotation
    if any(k in n for k in [
        "twist", "pallof", "woodchopper", "rotation"
    ]):
        return "core_rotational"

    # Cardio machines
    if any(k in n for k in [
        "treadmill", "elliptical", "bike", "cycling", "rowing"
    ]):
        return "cardio_machine"

    # Functional cardio / conditioning
    if any(k in n for k in [
        "burpee", "battle rope", "rope", "slam", "sled", "air bike", "step", "jump"
    ]):
        return "cardio_functional"

    # Mobility / yoga / rehab
    if any(k in n for k in [
        "stretch", "mobility", "pose", "pigeon", "cat", "cow", "child", "yoga"
    ]):
        return "mobility"

    # Balance / stability
    if any(k in n for k in [
        "balance", "bosu", "stability", "single-leg", "unstable"
    ]):
        return "stability"

    # Strength / hypertrophy (default)
    return "strength"


def infer_primary_muscles(name: str):
    family = infer_semantic_family(name)
    n = name.lower()

    if family == "weightlifting":
        return ["Glutes", "Hamstrings", "Quadriceps", "Lower Back", "Upper Back", "Core"]

    if family == "carry":
        return ["Core", "Forearms", "Upper Back"]

    if family == "core_rotational":
        return ["Core", "Obliques"]

    if family == "cardio_machine":
        return ["Quadriceps", "Calves"]

    if family == "cardio_functional":
        return ["Full Body"]

    if family == "mobility":
        return ["Core"]

    if family == "stability":
        return ["Core"]

    # strength (fallback con keywords suaves)
    if any(k in n for k in ["deadlift", "hip"]):
        return ["Hamstrings", "Glutes"]
    if any(k in n for k in ["squat", "lunge", "leg"]):
        return ["Quadriceps", "Glutes"]
    if any(k in n for k in ["overhead", "raise"]):
        return ["Deltoids", "Lats"]
    if any(k in n for k in ["bench", "push", "chest", "press"]):
        return ["Chest", "Triceps"]
    if any(k in n for k in ["row", "pull", "lat"]):
        return ["Lats", "Upper Back"]
    if any(k in n for k in ["curl"]):
        return ["Biceps"]
    if any(k in n for k in ["triceps", "dip"]):
        return ["Triceps"]
    if any(k in n for k in ["calf"]):
        return ["Calves"]
    if any(k in n for k in ["glute"]):
        return ["Glutes"]

    return ["Core"]


def infer_equipment(name: str):
    n = name.lower()
    family = infer_semantic_family(name)

    # Family-based inference (highest priority)

    if family == "weightlifting":
        return "Barbell"

    if family == "carry":
        # assume farmer / suitcase carry
        if any(k in n for k in ["kettlebell", "kb"]):
            return "Kettlebell"
        if any(k in n for k in ["dumbbell", "db"]):
            return "Dumbbell"
        return "Dumbbell"

    if family == "cardio_machine":
        if "elliptical" in n:
            return "Elliptical"
        if "row" in n:
            return "Rowing Machine"
        if "bike" in n or "cycle" in n:
            return "Cycling (Stationary)"
        if "box" in n or "step" in n:
            return "Box/Stepper"
        
        return "Treadmill"

    if family == "cardio_functional":
        if "rope" in n:
            return "Battle Ropes"
        if "sled" in n:
            return "Sled"
        return "None"

    if family == "mobility":
        return "None"

    if family == "stability":
        return "None"

    # Keyword-based inference (fallback)

    if any(k in n for k in ["barbell", "bench", "squat", "deadlift", "press", "clean", "snatch"]):
        return "Barbell"

    if any(k in n for k in ["dumbbell", "db "]):
        return "Dumbbell"

    if any(k in n for k in ["kettlebell", "kb "]):
        return "Kettlebell"

    if any(k in n for k in ["cable", "pulldown"]):
        return "Cable Machine"

    if any(k in n for k in ["trx", "suspension"]):
        return "TRX"

    if any(k in n for k in ["band", "resistance"]):
        return "Resistance Band"

    if any(k in n for k in ["pull-up", "chin-up"]):
        return "Pull-up Bar"

    if any(k in n for k in ["plank", "push up", "burpee", "lunge", "jump"]):
        return "None"

    # Safe fallback
    return "None"

def infer_type(name: str):
    family = infer_semantic_family(name)

    if family == "weightlifting":
        return "Power"
    if family in ["cardio_machine", "cardio_functional"]:
        return "Cardio"
    if family == "mobility":
        return "Mobility"
    if family == "stability":
        return "Stability"
    if family == "core_rotational":
        return "Stability"

    return "Strength"


def infer_intensity(name: str):
    family = infer_semantic_family(name)

    if family in ["mobility"]:
        return "Low"
    if family in ["cardio_machine", "cardio_functional"]:
        return "Moderate"
    if family in ["weightlifting"]:
        return "High"

    return "Moderate"


def infer_goals(name: str):
    family = infer_semantic_family(name)

    if family == "weightlifting":
        return ["Power", "Strength"]
    if family == "strength":
        return ["Strength", "Hypertrophy"]
    if family in ["cardio_machine", "cardio_functional"]:
        return ["Fat Loss", "Endurance"]
    if family == "mobility":
        return ["Mobility", "Rehab"]
    if family == "stability":
        return ["Rehab"]

    return ["Strength"]


# Build nodes and edges

nodes = []
edges = []

node_id = 0
id_map = {}  # name -> id

def add_node(name, ntype, attrs=None):
    global node_id
    if name in id_map:
        return id_map[name]
    nid = node_id
    id_map[name] = nid
    nodes.append({"node_id": nid, "name": name, "type": ntype, "attrs": attrs or {}})
    node_id += 1
    return nid

# add muscle nodes
for m in muscle_groups:
    add_node(m, "MuscleGroup")

# add equipment nodes
for e in equipment_list:
    add_node(e, "Equipment")

# add types, intensities, goals
for t in exercise_types:
    add_node(t, "ExerciseType")
for it in intensities:
    add_node(it, "IntensityLevel")
for g in exercise_goals:
    add_node(g, "ExerciseGoal")

# add exercise nodes and edges
variation_map = {}  # map exercise -> canonical exercise for isVariationOf
for ex in tqdm(exercises, desc="Adding exercises"):
    attrs = {}
    primary_m = infer_primary_muscles(ex)
    equip = infer_equipment(ex)
    etype = infer_type(ex)
    intensity = infer_intensity(ex)
    goals = infer_goals(ex)
    attrs["primary_muscles"] = primary_m
    attrs["equipment"] = equip
    attrs["exercise_type"] = etype
    attrs["intensity"] = intensity
    attrs["goals"] = goals
    nid = add_node(ex, "Exercise", attrs)
    # add relations: targets to primary muscles
    for m in primary_m:
        if m not in id_map:
            add_node(m, "MuscleGroup")
        edges.append({"source": nid, "target": id_map[m], "rel": "targets"})
    # equipment relation
    if equip not in id_map:
        add_node(equip, "Equipment")
    edges.append({"source": nid, "target": id_map[equip], "rel": "requires"})
    # type
    edges.append({"source": nid, "target": id_map[etype], "rel": "hasType"})
    # intensity
    edges.append({"source": nid, "target": id_map[intensity], "rel": "hasIntensity"})
    # goals
    for g in goals:
        edges.append({"source": nid, "target": id_map[g], "rel": "supportsGoal"})

# Add some isVariationOf edges by heuristics: if name contains 'Incline', 'Decline', 'Dumbbell' vs 'Barbell' variants
for name, nid in list(id_map.items()):
    if isinstance(name, str) and "Incline" in name:
        base = name.replace("Incline ","")
        if base in id_map:
            edges.append({"source": nid, "target": id_map[base], "rel": "isVariationOf"})
    if isinstance(name, str) and "Decline" in name:
        base = name.replace("Decline ","")
        if base in id_map:
            edges.append({"source": nid, "target": id_map[base], "rel": "isVariationOf"})
    if isinstance(name, str) and "Dumbbell" in name:
        base = name.replace("Dumbbell ","").strip()
        if base in id_map:
            edges.append({"source": nid, "target": id_map[base], "rel": "isVariationOf"})

# Create some recruit relations (secondary muscles) using simple rule: add core or nearby muscles
for n in nodes:
    if n["type"] == "Exercise":
        pid = n["node_id"]
        # if they target legs, add core recruit
        pms = n["attrs"].get("primary_muscles",[])
        if any(x in ["Quadriceps","Glutes","Hamstrings"] for x in pms):
            for m in ["Core"]:
                edges.append({"source": pid, "target": id_map[m], "rel": "recruits"})
        # if chest exercise, recruit triceps/shoulder
        if any(x in ["Chest"] for x in pms):
            for m in ["Triceps","Deltoids Anterior"]:
                if m in id_map:
                    edges.append({"source": pid, "target": id_map[m], "rel": "recruits"})

# remove duplicates edges
seen = set()
unique_edges = []
for e in edges:
    key = (e["source"], e["target"], e["rel"])
    if key not in seen:
        seen.add(key)
        unique_edges.append(e)
edges = unique_edges


# Export nodes.csv and edges.csv

nodes_df = pd.DataFrame(nodes)
# expand attrs to JSON string for CSV
nodes_df["attrs_json"] = nodes_df["attrs"].apply(lambda x: json.dumps(x, ensure_ascii=False))
nodes_df_out = nodes_df[["node_id","name","type","attrs_json"]]
nodes_df_out.to_csv(os.path.join(output_dir,"nodes.csv"), index=False, encoding="utf-8-sig")

edges_df = pd.DataFrame(edges)
edges_df.to_csv(os.path.join(output_dir,"edges.csv"), index=False, encoding="utf-8-sig")

# mapping
with open(os.path.join(output_dir,"mapping.json"), "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)

# summary
summary = {
    "num_nodes": len(nodes),
    "num_edges": len(edges),
    "node_types": pd.Series([n["type"] for n in nodes]).value_counts().to_dict(),
    "edge_types": pd.Series([e["rel"] for e in edges]).value_counts().to_dict()
}
with open(os.path.join(output_dir,"summary.txt"), "w", encoding="utf-8") as f:
    f.write(json.dumps(summary, indent=2, ensure_ascii=False))

print("Exported CSVs to", output_dir)
print("Summary:", summary)


# Build PyG graph if available

if pyg_available:
    print("Building PyG Data object...")
    # build node features: one-hot for types + simple name embedding fallback (length)
    node_types = sorted(set(n["type"] for n in nodes))
    type_to_idx = {t:i for i,t in enumerate(node_types)}
    X = []
    for n in nodes:
        onehot = [0]*len(node_types)
        onehot[type_to_idx[n["type"]]] = 1
        # simple numeric features: name length, number of attrs keys
        name_len = len(n["name"])
        attrs_count = len(n["attrs"].keys())
        numeric = [name_len/50.0, attrs_count/10.0]
        feat = onehot + numeric
        X.append(feat)
    x = torch.tensor(X, dtype=torch.float)

    # build edge_index (bidirectional)
    src = [e["source"] for e in edges]
    tgt = [e["target"] for e in edges]
    edge_index = torch.tensor([src+ tgt, tgt + src], dtype=torch.long)
    # encode edge types as ids
    rel_types = sorted(set(e["rel"] for e in edges))
    rel_to_idx = {r:i for i,r in enumerate(rel_types)}
    edge_attr = torch.tensor([rel_to_idx[e["rel"]] for e in edges] + [rel_to_idx[e["rel"]] for e in edges], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_attr
    )

    torch.save(data, os.path.join(output_dir,"graph_data.pt"))

    #data = Data(x=x, edge_index=edge_index)
    #torch.save(data, os.path.join(output_dir,"graph_data.pt"))
    torch.save(x, os.path.join(output_dir,"node_features.pt"))
    with open(os.path.join(output_dir,"edge_rel_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(rel_to_idx, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir,"node_type_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(type_to_idx, f, ensure_ascii=False, indent=2)
    print("Saved PyG graph_data.pt and node_features.pt")

else:
    print("PyG not available; resume by installing torch_geometric to build .pt objects.")

print("Done.")
