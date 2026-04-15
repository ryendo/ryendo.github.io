"""Rebuild public/files/eigenfunctions/data/index.json from the per-domain
JSON files currently on disk + the DOMAINS / FAMILIES definitions in the
main script. Does not recompute any FEM data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_eigenfunctions import DOMAINS, FAMILIES, OUT_DATA, _first_few

index = []
for spec in DOMAINS:
    p = OUT_DATA / f"{spec.id}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    index.append({
        "id": spec.id,
        "nameEn": spec.name_en,
        "nameJa": spec.name_ja,
        "descriptionEn": spec.description_en,
        "descriptionJa": spec.description_ja,
        "category": spec.category,
        "reference": spec.reference,
        "familyId": spec.family_id,
        "familyParam": spec.family_param,
        "mesh": d["mesh"],
        "firstFew": {bc: _first_few(d["boundaries"][bc]) for bc in d["boundaries"]},
    })

individuals = [d for d in index if d["familyId"] is None]
family_entries = []
for fam in FAMILIES:
    ms = [d for d in index if d["familyId"] == fam.id]
    if not ms:
        continue
    family_entries.append({
        "id": fam.id,
        "nameEn": fam.name_en,
        "nameJa": fam.name_ja,
        "descriptionEn": fam.description_en,
        "descriptionJa": fam.description_ja,
        "category": fam.category,
        "param": fam.param,
        "paramJa": fam.param_ja,
        "paramValues": list(fam.param_values),
        "memberIds": [d["id"] for d in ms],
        "reference": fam.reference,
    })

# Also read the analytic-families companion file (from compute_analytic.py)
# and append them alongside the FEM-computed families.
analytic_path = OUT_DATA / "_analytic_families.json"
if analytic_path.exists():
    analytic_families = json.loads(analytic_path.read_text())
    for fam in analytic_families:
        # Load each analytic-domain JSON and splice into `index`.
        for mid in fam["memberIds"]:
            p = OUT_DATA / f"{mid}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            index.append({
                "id": d["id"],
                "nameEn": d["nameEn"],
                "nameJa": d["nameJa"],
                "descriptionEn": d["descriptionEn"],
                "descriptionJa": d["descriptionJa"],
                "category": d["category"],
                "reference": d.get("reference", ""),
                "familyId": d.get("familyId"),
                "familyParam": d.get("familyParam"),
                "mesh": d.get("mesh", {"vertices": 0, "triangles": 0, "element": "analytic"}),
                "analytic": d.get("analytic", True),
                "dimension": d.get("dimension", "2d"),
                "firstFew": {bc: _first_few(d["boundaries"][bc]) for bc in d["boundaries"]},
            })
        # register family
        family_entries.append(fam)

(OUT_DATA / "index.json").write_text(
    json.dumps({"domains": index, "individuals": individuals, "families": family_entries},
               indent=2, ensure_ascii=False)
)
print(f"rebuilt index.json: {len(individuals)} individuals, {len(family_entries)} families, {len(index)} total")
