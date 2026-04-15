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

(OUT_DATA / "index.json").write_text(
    json.dumps({"domains": index, "individuals": individuals, "families": family_entries},
               indent=2, ensure_ascii=False)
)
print(f"rebuilt index.json: {len(individuals)} individuals, {len(family_entries)} families, {len(index)} total")
