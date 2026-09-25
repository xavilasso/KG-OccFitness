# KG-OccFitness

Construction of the **Fitness Knowledge Graph (FKG)**, a heterogeneous multi-relational knowledge
graph of physical exercise built from a public-domain exercise catalogue and anchored to a
reference anatomy ontology.

This repository produces the artefact described in:

> K. Avila, J. Lasso-Saldaña, J. Gálvez, O. Avalos. *Design and Construction of a Multi-Relational
> Fitness Knowledge Graph for Semantic Exercise Representation.* Mathematics, MDPI, 2026.
> DOI: `[PENDIENTE: DOI del artículo]`

The learning experiments that validate the graph live in a separate repository,
[fitOcc-rec-gnn](https://github.com/xavilasso/fitOcc-rec-gnn). The results reported in the article
were produced with release `[PENDIENTE: tag, p. ej. v1.0.0]` of this repository
(Zenodo DOI `[PENDIENTE]`) and release `[PENDIENTE]` of fitOcc-rec-gnn.

---

## What the graph contains

| | |
|---|---|
| Entities | **922** |
| Typed assertions | **8,598** |
| Relation types | **9** |
| Connected components | 1 (all 922 entities) |
| Schema violations | 0 |

**Entity types.** Exercise (873), MuscleGroup (17), Equipment (12), ExerciseGoal (7),
ExerciseType (5), IntensityLevel (3), ForceType (3), MechanicType (2).

**Relations**, with domain → range and number of assertions:

| Relation | Domain → Range | Assertions | Density |
|---|---|---:|---:|
| `recruits` | Exercise → MuscleGroup | 1,701 | 0.167 |
| `supportsGoal` | Exercise → ExerciseGoal | 1,647 | 0.270 |
| `targets` | Exercise → MuscleGroup | 873 | 0.059 |
| `requires` | Exercise → Equipment | 873 | 0.083 |
| `hasType` | Exercise → ExerciseType | 873 | 0.200 |
| `hasIntensity` | Exercise → IntensityLevel | 873 | 0.333 |
| `hasForce` | Exercise → ForceType | 844 | 0.333 |
| `hasMechanic` | Exercise → MechanicType | 786 | 0.500 |
| `isVariationOf` | Exercise → Exercise | 128 | 0.015 |

Density is the bipartite density of the relation, that is, the proportion of admissible
subject–object pairs that are actually asserted: `|E_r| / (|dom(r)| · |ran(r)|)`, where
`dom(r)` and `ran(r)` are the entities that actually appear as subject and object of `r`.

Every assertion is checked against a declared domain and range before the graph is written. By
default a violation aborts the construction (`--allow-warnings` downgrades it to a warning).

---

## Repository layout

```
build_fitkg_lite_v3.py      construction pipeline (the one that produces the published graph)
map_muscles_to_uberon.py    anchors the muscle vocabulary to Uberon via the EBI OLS4 API
muscle_uberon_mapping.csv   resulting mapping, with match type and expert confirmation
data/exercises.json         source catalogue (Free Exercise Database, 873 exercises)
fitkg_v3_output/            the published graph
requirements.txt
LICENSE                     MIT, for the code
LICENSE-DATA                CC BY 4.0, for fitkg_v3_output/ and muscle_uberon_mapping.csv
```

### `fitkg_v3_output/`

| File | Contents |
|---|---|
| `nodes.csv` | `node_id, name, type, attrs_json` — one row per entity. `attrs_json` holds `source_id`, `force`, `mechanic`, `level` and `curated` for exercises |
| `edges.csv` | `source, target, rel` — one row per assertion, directed, no mirrors, no inverses |
| `summary.json` | entity and relation counts |
| `graph_data.pt` | PyTorch Geometric `Data` object: `x` (922 × 8, one-hot of the entity type), `edge_index` (2 × 8,598), `edge_type` (8,598) |
| `edge_rel_mapping.json` | relation name → integer id used in `edge_type` (alphabetical order) |
| `node_type_mapping.json` | entity type → column of `x` (alphabetical order) |
| `isvariationof_para_validar.csv` | the 121 family-derived `isVariationOf` candidates, with an `expert_verdict` column |

CSV files are written as UTF-8 with BOM (`utf-8-sig`) so that they open correctly in Excel.

`graph_data.pt` is the input of the experiments in fitOcc-rec-gnn. Its `node_id` order is the
row order of `nodes.csv`, and `edge_type` uses the ids of `edge_rel_mapping.json`.

SHA-256 of the published files, for verification:

```
04c810e46c9f8f895f2331700b3d15a2aedf08db67ef5cdf16c6840567542d0e  graph_data.pt
f5c4462c6210a60c6be8958cdde8fd1e40debe398fd6077426ce3cd68a358f35  nodes.csv
87dbd9d26ee992657c6c997590c90af11fb19c175428a921279ad4e05edee553  edges.csv
3fbd57bcfe2818d33298d7b7697c8a3c479fb64cacd213db202c15388709102f  edge_rel_mapping.json
1a4e68f18c89e4abb9234466f3adf790fdd644c13a18ea34551451f27e53b2e9  summary.json
2cbb723eada5b819dace8d0753066790f25b12c4a1fc814141d7ae6638842e81  data/exercises.json
```

---

## Reproducing the graph

Tested with Python 3.11.9 on Windows 11, CPU only.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python build_fitkg_lite_v3.py --input data/exercises.json      # writes fitkg_v3_output/
```

`--input` is required. Other options: `--out` (default `fitkg_v3_output`), `--no-goals` (omit
`supportsGoal`), `--curated` (JSON `{exercise name: [primary muscles]}` that overrides the
catalogue; not used for the published graph, so `curated` is `false` for all 873 exercises).

The construction is deterministic. Rebuilding from `data/exercises.json` with the versions in
`requirements.txt` reproduces every file of `fitkg_v3_output/` byte for byte, `graph_data.pt`
included. If PyTorch Geometric is not installed the CSV and JSON files are still written, but
`graph_data.pt`, `edge_rel_mapping.json` and `node_type_mapping.json` are not.

### Uberon mapping

```bash
python map_muscles_to_uberon.py --nodes fitkg_v3_output/nodes.csv --out muscle_uberon_new.csv
```

The script queries the EBI OLS4 API, so it needs network access and its result depends on the
state of Uberon at the time of the query. **Do not write over `muscle_uberon_mapping.csv`**: the
published file carries the expert confirmation in the column `expert_confirmed`, and the script
writes that column empty. The mapping is distributed as a separate table; the entities of
`nodes.csv` do not carry the Uberon identifier as an attribute.

---

## Data sources

| Source | Licence | Role |
|---|---|---|
| [Free Exercise Database](https://github.com/yuhonas/free-exercise-db) | Unlicense (public domain) | exercise catalogue, `dist/exercises.json` at commit `[PENDIENTE: SHA de free-exercise-db]` |
| [Uberon](https://obofoundry.org/ontology/uberon.html) | CC BY 3.0 | anatomical identifiers for the muscle vocabulary, retrieved through OLS4 on `[PENDIENTE: fecha]` |
| [wger](https://github.com/wger-project/wger) | AGPL-3.0 | consulted as a complementary open resource; no wger content is included in this repository |

### How the source fields become relations

| Source field | Relation | Coverage |
|---|---|---:|
| `primaryMuscles` | `targets` | 873 / 873 |
| `secondaryMuscles` | `recruits` | 598 / 873 |
| `equipment` | `requires` | 873 / 873 |
| `category` | `hasType` and `supportsGoal` | 873 / 873 |
| `level` | `hasIntensity` | 873 / 873 |
| `force` | `hasForce` | 844 / 873 |
| `mechanic` | `hasMechanic` | 786 / 873 |
| `name` (derived) | `isVariationOf` | 126 / 873 as subject |

Two relations are derived rather than transcribed, under deterministic mappings:

```
hasIntensity   beginner → Low    intermediate → Moderate    expert → High

supportsGoal   strength                → {Strength, Hypertrophy}
               powerlifting            → {Strength}
               olympic weightlifting   → {Power, Strength}
               strongman               → {Power, Strength}
               cardio                  → {Endurance, Fat Loss}
               stretching              → {Mobility, Rehab}
               plyometrics             → {Power}
```

The `category` field also determines the `ExerciseType` entity, where the three weightlifting
modalities are grouped under a single type (`Power`). This is why the type vocabulary has five
entities while the source modalities are seven.

`isVariationOf` comes from two deterministic rules applied to exercise names:

1. **Suffix rule** (7 assertions): a name of the form `<base> - <qualifier>` or
   `Incline/Decline <base>` points to `<base>` when that exercise exists.
2. **Movement family** (121 assertions): after removing equipment and posture qualifiers from
   the start of the name, exercises that share the same movement core and at least one primary
   muscle point to the member with the shortest name. These are the candidates listed in
   `isvariationof_para_validar.csv`.

Both sets are included in the published graph (7 + 121 = 128). 126 exercises appear as subject
and 190 exercises (21.8 %) are touched by the relation.

Three source fields are incompletely populated and the gaps are inherited rather than imputed:
`force` is absent for 29 exercises, `mechanic` for 87, and `secondaryMuscles` for 275.

---

## Expert review

Two sets of assertions were submitted to a certified strength and conditioning specialist:

1. The mapping between the colloquial muscle labels of the catalogue and their anatomical
   referents, with the type of correspondence declared as exact, narrow or broad
   (`muscle_uberon_mapping.csv`). All 17 rows are confirmed (`expert_confirmed = yes`).
2. The candidate `isVariationOf` assertions, each presented as a variant, its proposed base
   exercise and the movement family grouping them (`isvariationof_para_validar.csv`).
   `[PENDIENTE: la columna expert_verdict está vacía en las 121 filas. Rellenarla con el veredicto
   del experto antes del release, o describir aquí el resultado de la revisión y si alguna arista
   rechazada se retiró del grafo.]`

---

## Citation

```bibtex
@article{avila2026fkg,
  title   = {Design and Construction of a Multi-Relational Fitness Knowledge Graph
             for Semantic Exercise Representation},
  author  = {Avila, Karla and Lasso-Salda{\~n}a, Javier and G{\'a}lvez, Jorge and Avalos, Omar},
  journal = {Mathematics},
  year    = {2026},
  doi     = {[PENDIENTE]}
}
```

## Licence

Code is released under the MIT Licence (`LICENSE`). The derived knowledge graph in
`fitkg_v3_output/` and `muscle_uberon_mapping.csv` are released under CC BY 4.0 (`LICENSE-DATA`).
The source catalogue in `data/exercises.json` is in the public domain (Unlicense). The Uberon
identifiers and labels in `muscle_uberon_mapping.csv` come from Uberon under CC BY 3.0, which
permits redistribution with attribution; reuse of the derived resource therefore requires
attribution to this work and to Uberon.
