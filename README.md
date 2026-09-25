# KG-OccFitness

Construction of the **Fitness Knowledge Graph (FKG)**, a heterogeneous multi-relational knowledge
graph of physical exercise built from a public-domain exercise catalogue and anchored to a
reference anatomy ontology.

This repository produces the artefact described in:

> K. Avila, J. Lasso-Saldaña, J. Gálvez, O. Avalos. *Design and Construction of a Multi-Relational
> Fitness Knowledge Graph for Semantic Exercise Representation.* Mathematics, MDPI. `[CONFIRMAR año/DOI]`

The learning experiments that validate the graph live in a separate repository,
[fitOcc-rec-gnn](https://github.com/xavilasso/fitOcc-rec-gnn).

---

## What the graph contains

| | |
|---|---|
| Entities | **922** |
| Typed assertions | **8,598** |
| Relation types | **9** |
| Connected components | 1 |
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
subject–object pairs that are actually asserted: `|E_r| / (|dom(r)| · |ran(r)|)`.

Every assertion is checked against a declared domain and range **before** the graph is written;
an assertion that does not comply aborts the construction instead of being stored.

---

## Repository layout

```
build_fitkg_lite_v3.py      construction pipeline (the one that produces the published graph)
map_muscles_to_uberon.py    anchors the muscle vocabulary to Uberon via the EBI OLS4 API
muscle_uberon_mapping.csv   resulting mapping, with match type and expert confirmation
data/                       source catalogue
fitkg_v3_output/            the published graph
requirements.txt
```

### `fitkg_v3_output/`

| File | Contents |
|---|---|
| `nodes.csv` | `node_id, name, type, attrs_json` — one row per entity |
| `edges.csv` | `source, rel, target` — one row per assertion, directed, no mirrors |
| `summary.json` | entity and relation counts |
| `isvariationof_para_validar.csv` | candidate variation assertions submitted to expert review |

---

## Reproducing the graph

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python build_fitkg_lite_v3.py            # writes fitkg_v3_output/
python map_muscles_to_uberon.py          # refreshes muscle_uberon_mapping.csv (needs network)
```

The construction is deterministic: the same source catalogue yields the same graph.

---

## Data sources

| Source | Licence | Role |
|---|---|---|
| [Free Exercise Database](https://github.com/yuhonas/free-exercise-db) | Unlicense (public domain) | exercise catalogue, commit `[CONFIRMAR SHA]` |
| [Uberon](https://obofoundry.org/ontology/uberon.html) | CC BY 3.0 | anatomical identifiers for the muscle vocabulary |
| [wger](https://github.com/wger-project/wger) | AGPL-3.0 | consulted as a complementary open resource |

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
| `name` (derived) | `isVariationOf` | 126 / 873 |

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

Three source fields are incompletely populated and the gaps are inherited rather than imputed:
`force` is absent for 29 exercises, `mechanic` for 87, and `secondaryMuscles` for 275.

---

## Expert review

Two sets of assertions were submitted to a certified strength and conditioning specialist:

1. The mapping between the colloquial muscle labels of the catalogue and their anatomical
   referents, with the type of correspondence declared as exact, narrow or broad
   (`muscle_uberon_mapping.csv`).
2. The candidate `isVariationOf` assertions, each presented as a variant, its proposed base
   exercise and the movement family grouping them (`isvariationof_para_validar.csv`).

---

## Citation

```bibtex
@article{avila2026fkg,
  title   = {Design and Construction of a Multi-Relational Fitness Knowledge Graph
             for Semantic Exercise Representation},
  author  = {Avila, Karla and Lasso-Salda{\~n}a, Javier and G{\'a}lvez, Jorge and Avalos, Omar},
  journal = {Mathematics},
  year    = {2026},
  doi     = {[CONFIRMAR]}
}
```

## Licence

Code released under the MIT Licence. The derived knowledge graph in `fitkg_v3_output/` is released
under CC BY 4.0. The source catalogue is in the public domain (Unlicense) and the Uberon
identifiers are CC BY 3.0, so redistribution of the derived resource is unrestricted.
