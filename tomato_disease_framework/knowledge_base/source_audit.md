# Phase 10 candidate-source inventory

**Inventory date:** 2026-10-24
**Scope:** Candidate references associated with background and reported-symptom drafts for the ten PlantVillage tomato class IDs.
**Verification status:** The source contents and claims have not been independently re-checked in this review. The source URLs and statements are retained as user-provided curated data, but all nine source records are marked `unverified` in the JSON. Listing a URL or passing schema tests does not verify that its contents support the associated claim. No source authorizes recommendations.

## Candidate sources (not verified in this review)

| Class | Source | Why used | Scope limitation |
|---|---|---|---|
| Bacterial spot | [TNAU — Bacterial leaf spot](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_11.html) | Tamil Nadu-specific crop-protection page with symptoms, causal organism, and spread | TNAU uses the name “bacterial leaf spot”; pathogen taxonomy/nomenclature may vary among references |
| Early blight | [TNAU — Early Blight](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html) | Tamil Nadu-specific description of Alternaria solani and classic concentric lesions | Not a substitute for field diagnosis |
| Late blight | [TNAU — Late blight](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html) | Tamil Nadu-specific description of Phytophthora infestans and symptoms | Conditions and local advisories vary by season and district |
| Leaf mold | [University of Minnesota Extension — Tomato leaf mold](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold) | Clear symptoms and causal-organism description | US extension source; greenhouse/high-tunnel context may not represent all Tamil Nadu production systems |
| Septoria leaf spot | [TNAU — Septoria Leaf Spot](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_4.html) | Tamil Nadu-specific symptoms and disease cycle | Visual symptoms overlap with other leaf-spot diseases |
| Spider mites | [TNAU — Red spider mite: Tetranychus spp.](https://www.agritech.tnau.ac.in/crop_protection/tomato/tomato_7.html) | Tamil Nadu-specific tomato pest page | TNAU describes *Tetranychus* spp.; it does **not** prove the dataset class is specifically *T. urticae* in every image |
| Target spot | [UF/IFAS Extension — Target Spot of Tomato in Florida](https://ask.ifas.ufl.edu/publication/PP351) | University extension publication describing *Corynespora cassiicola*, symptoms, and cultural context | Florida-specific management context; no pesticide recommendations imported into this KB |
| Tomato yellow leaf curl virus | [TNAU — Leaf curl](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_6.html) | Tamil Nadu-specific symptoms and whitefly association | TNAU’s “tomato leaf curl virus” label is broader than a molecular confirmation of the PlantVillage TYLCV label |
| Tomato mosaic virus | [Utah State University Extension — Tobacco Mosaic Virus & Tomato Mosaic Virus](https://extension.usu.edu/planthealth/ipm/notes_ag/veg-TMV-ToMV) | Extension source covering ToMV symptoms and transmission | US source; symptoms can overlap with other viral and non-viral disorders |
| Healthy | No disease source | No disease claim is appropriate for the healthy class | The classifier's “healthy” label is not a guarantee that the whole plant is disease-free |

## Verification and publication dates

- The candidate pages and linked claims require direct content review before being marked verified. `verified_on` and `verification_method` are empty for all records until that review occurs.
- `publication_date` is left blank because a publication/revision date has not been verified. The inventory date is not represented as the source publication date.
- No source is cited for the healthy class.

## Why management guidance remains empty

`management_guidance` intentionally remains empty for every record. Source contents have not been verified in this review, and no management claims have been added. This keeps the existing policy fail-closed: no recommendations are emitted until sources and any guidance are checked for provenance, relevance, sufficiency, and conflicts.

## Research limitations

1. These are source-backed descriptions of reported disease/pest features, not proof that symptoms in a particular image are caused by the predicted class.
2. Several classes have overlapping symptoms. The UI should retain uncertainty and recommend confirmation by a local plant pathology/extension service when the result would drive an intervention.
3. TNAU pages are preferred for Tamil Nadu context where a relevant page was found. Non-local sources are clearly identified above.
4. No severity, progression, lesion segmentation, or field-generalization claim is added here.
