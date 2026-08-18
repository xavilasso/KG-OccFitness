#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ancla el vocabulario de grupos musculares del FKG a la ontologia Uberon.

Por que: los 17 grupos musculares del grafo son strings coloquiales del dataset
("lats", "middle back", "traps"). Ancladas a identificadores Uberon dejan de ser
vocabulario ad hoc y pasan a ser entidades interoperables: cualquier otro recurso
biomedico que use Uberon puede alinearse con el grafo sin renegociar los nombres.
Uberon es CC BY 3.0, asi que el mapeo se puede publicar junto al grafo.

Que hace este script:
  1. Toma cada termino coloquial y su equivalente anatomico propuesto (tabla de
     abajo, revisada a mano: ahi esta la decision, no en el codigo).
  2. Consulta la API publica de OLS4 del EBI y trae los candidatos Uberon.
  3. Escribe muscle_uberon_mapping.csv con el candidato, el tipo de
     correspondencia y una columna vacia para el visto bueno del experto.

El tipo de correspondencia importa y es lo que un revisor va a mirar:
  exact  - el termino coloquial y el termino Uberon designan lo mismo
  narrow - el termino Uberon es MAS especifico que el coloquial
           (p. ej. "chest" se ancla a pectoralis major, que es una parte)
  broad  - el termino Uberon es MAS general que el coloquial

Uso:
    pip install requests
    python map_muscles_to_uberon.py --nodes fitkg_v3_output/nodes.csv
"""

import argparse
import csv
import sys
import time

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("Falta requests:  pip install requests")

OLS = "https://www.ebi.ac.uk/ols4/api/search"

# Termino coloquial -> (termino anatomico a buscar, tipo de correspondencia, nota)
# Esta tabla es la parte que requiere criterio; la revision del experto deberia
# concentrarse aqui, no en los identificadores que devuelva el servicio.
ANATOMY = {
    "quadriceps":  ("quadriceps femoris", "exact",  ""),
    "hamstrings":  ("hamstring muscle", "exact",
                    "grupo: biceps femoris, semitendinosus, semimembranosus"),
    "glutes":      ("gluteal muscle", "exact",
                    "grupo: gluteus maximus, medius, minimus"),
    "calves":      ("triceps surae muscle", "narrow",
                    "'calves' incluye tambien tibialis posterior y peroneos"),
    "chest":       ("pectoralis major", "narrow",
                    "'chest' incluye pectoralis minor y serratus anterior"),
    "lats":        ("latissimus dorsi", "exact",  ""),
    "traps":       ("trapezius", "exact",  ""),
    "shoulders":   ("deltoid", "narrow",
                    "'shoulders' incluye el manguito rotador; el dataset lo usa "
                    "sobre todo como deltoides"),
    "biceps":      ("biceps brachii", "exact",
                    "UBERON:0001507 verificado manualmente"),
    "triceps":     ("triceps brachii", "exact",  ""),
    "forearms":    ("muscle of forearm", "broad",
                    "region, no musculo unico"),
    "abdominals":  ("rectus abdominis", "narrow",
                    "'abdominals' incluye transverso y oblicuos"),
    "lower back":  ("erector spinae", "narrow",
                    "'lower back' incluye multifidus y quadratus lumborum"),
    "middle back": ("rhomboid muscle", "narrow",
                    "'middle back' incluye trapecio medio y romboides"),
    "adductors":   ("adductor muscle of thigh", "exact",  ""),
    "abductors":   ("abductor muscle", "broad",
                    "en el dataset se refiere a los abductores de cadera"),
    "neck":        ("neck muscle", "broad",  "region, no musculo unico"),
}


def query_ols(term, rows=3, pause=1.0):
    try:
        r = requests.get(OLS, params={"q": term, "ontology": "uberon", "rows": rows},
                         timeout=20)
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
    except Exception as exc:                       # noqa: BLE001
        print(f"   ! fallo consultando '{term}': {exc}")
        return []
    finally:
        time.sleep(pause)
    return [{"id": d.get("short_form", "").replace("_", ":"),
             "label": d.get("label", ""),
             "iri": d.get("iri", "")} for d in docs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", default="fitkg_v3_output/nodes.csv")
    p.add_argument("--out", default="muscle_uberon_mapping.csv")
    p.add_argument("--pause", type=float, default=1.0,
                   help="segundos entre consultas; subelo si te devuelve 429")
    a = p.parse_args()

    nodes = pd.read_csv(a.nodes)
    muscles = sorted(nodes.loc[nodes["type"] == "MuscleGroup", "name"].astype(str))
    print(f"grupos musculares en el grafo: {len(muscles)}\n")

    rows = []
    for m in muscles:
        key = m.strip().lower()
        term, match, note = ANATOMY.get(key, (m, "unreviewed",
                                              "sin equivalente anatomico propuesto"))
        cands = query_ols(term)
        best = cands[0] if cands else {"id": "", "label": "", "iri": ""}
        alts = "; ".join(f"{c['id']} {c['label']}" for c in cands[1:])
        print(f"{m:14s} -> {term:26s} {best['id']:16s} {best['label']}")
        rows.append({
            "muscle_group": m,
            "anatomical_term": term,
            "uberon_id": best["id"],
            "uberon_label": best["label"],
            "uberon_iri": best["iri"],
            "match_type": match,
            "note": note,
            "alternatives": alts,
            "expert_confirmed": "",
        })

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nEscrito {a.out}")
    print("Revisa 'match_type' y 'uberon_label' con el experto, llena "
          "'expert_confirmed' y vuelvemelo a pasar: con eso los nodos de musculo "
          "pasan a llevar su identificador Uberon como atributo.")


if __name__ == "__main__":
    main()