# Phase 10A source-content audit

**Review date:** 2026-10-24
**Scope:** The nine disease-source records currently present in `tomato_diseases.json`. The healthy class has no disease source and is not included in the source count.
**Method:** Retrieved the exact stored publisher URLs with the available page-fetch tool and, when that failed, retried selected pages with Python HTTPS requests. For two TNAU pages, content was read only after disabling certificate verification; those pages are not treated as authenticated source content. Claims below are compared with returned page text, not independently corroborated against the cited references or reviewed by a human domain specialist.

## Status and policy

| Audit classification | Count |
|---|---:|
| Fully verified | 0 |
| Partially supported by retrieved content | 6 |
| Unverified (content inaccessible or publisher identity unauthenticated) | 3 |
| Contradicted | 0 |

“Partially supported” describes the current claim/content comparison and does **not** mean a source is verified in the production knowledge base. All nine JSON source records therefore remain `unverified`; `verified_on` and `verification_method` remain empty. No management guidance has been added. The recommendation policy remains fail-closed.

This report records an assistant-performed, tool-assisted content comparison. It is not a claim of completed human manual verification, independent scientific corroboration, or confirmation of disease in an individual image. Source pages may also contain management material; no such material was imported.

## Individual source records

### 1. Bacterial spot — `tnau_tomato_bacterial_leaf_spot`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_11.html
- **Publisher/type and page heading:** Tamil Nadu Agricultural University Agritech Portal; “Bacterial leaf spot” (institutional crop-protection page).
- **Access/review:** Exact URL returned readable page text over HTTPS on 2026-10-24. Relevant disease, causal-organism, and spread sections were reviewed.
- **Claims checked/evidence:** “Field Diagnostic Symptoms” describes small brown water-soaked circular leaf spots with yellow halos and water-soaked green-fruit spots becoming irregular, light brown, sunken, and scabby. “Causal organism” names *Xanthomonas axonopodis* pv. *vesicatoria*. “Survival and Mode of Spread” lists infected debris, infected plant debris and seed as primary spread, and rain splash as secondary spread.
- **Assessment:** The JSON’s general Xanthomonas association, affected leaves/green fruit, debris/seed spread, and listed symptoms are supported by the retrieved sections. The page’s pathogen nomenclature has not been checked against current taxonomy or its cited underlying sources.
- **Final audit status:** **Partially supported.** Content matches the claims reviewed, but the source’s underlying references and taxonomic currency were not independently checked.
- **Limitations and next action:** TNAU content is relevant to its agricultural context; it does not diagnose a PlantVillage image. Have a plant-pathology reviewer check the complete page, nomenclature, and cited references before considering source verification.

### 2. Early blight — `tnau_tomato_early_blight`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html
- **Publisher/type and page heading:** Tamil Nadu Agricultural University Agritech Portal; “Early Blight: *Alternaria solani*” (institutional crop-protection page).
- **Access/review:** Exact URL returned readable page text over HTTPS with certificate verification on 2026-10-24. Relevant sections were reviewed.
- **Claims checked/evidence:** “Field Diagnostic Symptoms” describes brown spots with concentric bull’s-eye rings and yellow margin, and brown concentric rings on fruit. “Causal organism” names *Alternaria solani*. “Survival and Mode of Spread” lists infected plant debris and conidia carried by rain and wind.
- **Assessment:** The organism, lesion, fruit-ring, debris, rain, and wind statements in the JSON are supported by the retrieved sections. The page additionally mentions fruit infection through the calyx or stem attachment and infected seed; those details are not required by the current JSON claim.
- **Final audit status:** **Partially supported.** The claims reviewed match the exact page, but the entire publication and cited source material have not received independent review.
- **Limitations and next action:** A bull’s-eye pattern is not image-level confirmation or proof that the class is uniquely identified. Have a plant-pathology reviewer check the complete page and source references before changing verification metadata.

### 3. Late blight — `tnau_tomato_late_blight`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html
- **Publisher/type and page heading:** The URL is on the TNAU Agritech Portal; returned page heading “Late blight: *Phytophthora infestans*”.
- **Access/review:** The page returned HTTP 200 only when TLS certificate verification was disabled. A verified TLS request failed because the certificate was expired. The page text was read, but the publisher identity and integrity of that response could not be authenticated in this review.
- **Claims checked/evidence:** Returned text under “Symptoms”, “Fruit”, and “Survival and Mode of Spread” states water-soaked black lesions on leaves and stems, rapid leaf necrosis, white sporulation, dark-brown fruit lesions, soft rot/disintegration, *P. infestans*, infected-debris spread, and wind spread.
- **Assessment:** The returned text appears to support the JSON statements, including rapid necrosis and the fruit symptoms. The response is not accepted as authenticated evidence because the TLS certificate validation failed.
- **Final audit status:** **Unverified.**
- **Limitations and next action:** Reopen the exact page in a browser after its certificate is valid or obtain an authenticated official TNAU copy. Then re-check the full page and references. Do not treat the page text fetched with certificate checks disabled as verified evidence.

### 4. Leaf mold — `umn_tomato_leaf_mold`

- **Exact URL:** https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold
- **Publisher/type and page title:** The JSON record identifies University of Minnesota Extension and “Tomato leaf mold”; page title and publisher attribution were not confirmed from accessible page content.
- **Access/review:** Exact URL returned HTTP 403 from both page-fetch and direct HTTP clients on 2026-10-24. The page content was not checked.
- **Claims checked/evidence:** None. The organism and synonym list (*Passalora fulva*, *Fulvia fulva*, *Cladosporium fulvum*), high-humidity/greenhouse association, leaf-surface symptom contrast, and withering claims remain unchecked.
- **Assessment:** No claim is assessed as supported or contradicted by this audit.
- **Final audit status:** **Unverified.**
- **Limitations and next action:** A researcher must access the original page or an accessible copy hosted by University of Minnesota Extension and compare each claim, title, and publication/update metadata. Do not substitute another source without recording that substitution separately.

### 5. Septoria leaf spot — `tnau_tomato_septoria_leaf_spot`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_4.html
- **Publisher/type and page heading:** Tamil Nadu Agricultural University Agritech Portal; “Septoria Leaf Spot: *Septoria lycopersici*” (institutional crop-protection page).
- **Access/review:** Exact URL returned readable page text over HTTPS on 2026-10-24. Relevant symptom, organism, and spread sections were reviewed.
- **Claims checked/evidence:** “Symptoms” describes small round-to-irregular leaf spots with grey centers and dark margins, coalescence, and blighting. “Causal Organism” names *Septoria lycopersici*. “Survival and Mode of Spread” lists infected debris and conidia spread by rain splash or wind.
- **Assessment:** The JSON’s organism, debris/spread, spot appearance, and blighting statements are supported by the retrieved sections. The page also mentions defoliation and affected stems/flowers, which the JSON does not claim.
- **Final audit status:** **Partially supported.** The statements reviewed match the exact page, but cited underlying references and current taxonomy have not been independently checked.
- **Limitations and next action:** Lesion descriptions overlap with other conditions and do not establish an individual image’s diagnosis. Have a plant-pathology reviewer check the complete page and references before considering verification.

### 6. Spider mites — `tnau_tomato_red_spider_mite`

- **Exact URL:** https://www.agritech.tnau.ac.in/crop_protection/tomato/tomato_7.html
- **Publisher/type and page heading:** The JSON identifies TNAU Agritech Portal; returned page labels the pest “Red spider mite” and “*Tetranychus* spp.” (crop-protection page).
- **Access/review:** Exact URL returned HTTP 200, but TLS hostname validation failed because the certificate was not valid for `www.agritech.tnau.ac.in`. Page text was read with certificate verification disabled and is not treated as authenticated publisher content.
- **Claims checked/evidence:** Returned text under “Symptoms of damage” says affected leaves become reddish brown and bronzy; severe infestation includes silken webbing, withering/drying, and effects on flower and fruit formation. The returned page text names *Tetranychus* spp.
- **Assessment:** The page text supports the genus-level wording and the reported damage. It does not establish *T. urticae* or species identity for any individual PlantVillage image.
- **Final audit status:** **Unverified** because publisher identity was not authenticated, despite the text matching the cited page.
- **Limitations and next action:** Retry the exact URL after its TLS hostname/certificate issue is fixed or inspect an authenticated official TNAU page. Even then, the source’s genus-level label must not be promoted to species-level confirmation of dataset images.

### 7. Target spot — `uf_ifas_target_spot_tomato`

- **Exact URL:** https://ask.ifas.ufl.edu/publication/PP351
- **Publisher/type and page heading:** University of Florida Institute of Food and Agricultural Sciences Extension; publication PP351, “Target spot of tomato” (extension publication).
- **Access/review:** Exact URL returned page text over HTTPS on 2026-10-24. The disease description, leaf/fruit symptom, and canopy sections were reviewed.
- **Claims checked/evidence:** The opening description names *Corynespora cassiicola*, states occurrence in tropical/subtropical regions and greenhouse production, and notes that US distribution is largely confined to Florida. Symptom sections describe dark pinpoint lesions enlarging to light-brown/gray centers with dark concentric rings and sometimes diffuse yellowing, fruit flecks becoming deeply pitted, and disease developing in the inner canopy.
- **Assessment:** The JSON’s organism, broad pathogen occurrence, leaf and fruit descriptions, and inner-canopy statement are supported. The publication explicitly says leaf symptoms can be confused with other diseases and emphasizes Florida.
- **Final audit status:** **Partially supported.** Content supports the claims, but the Florida-centered source does not establish Tamil Nadu prevalence, epidemiology, or management applicability.
- **Limitations and next action:** Preserve the Florida context. Do not infer image-level diagnosis or generalize local conditions. Obtain local evidence before making Tamil Nadu-specific claims.

### 8. Tomato yellow leaf curl virus — `tnau_tomato_leaf_curl`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_6.html
- **Publisher/type and page heading:** Tamil Nadu Agricultural University Agritech Portal; “Leaf curl: Tomato leaf curl virus” (institutional crop-protection page).
- **Access/review:** Exact URL returned readable page text over HTTPS on 2026-10-24. Symptom, causal-organism, vector, and spread sections were reviewed.
- **Claims checked/evidence:** The page calls the causal organism “Tomato leaf curl virus (TLCV)” and names whiteflies, *Bemisia tabaci*, as vector. It lists severe stunting, downward rolling/crinkling, leathery/brittle older leaves, shortened internodes, and bushy appearance; it lists whitefly primary spread and movement of infected plants.
- **Assessment:** The symptom, vector, and spread statements are supported by the retrieved text. The page’s TLCV name does not establish the specific TYLCV identity in the PlantVillage class.
- **Final audit status:** **Partially supported.**
- **Limitations and next action:** Retain the JSON caveat that this is not strain/species-level confirmation. Obtain a source explicitly identifying TYLCV if virus-specific claims are required; visual symptoms alone do not confirm virus identity.

### 9. Tomato mosaic virus — `usu_tobacco_tomato_mosaic_virus`

- **Exact URL:** https://extension.usu.edu/planthealth/ipm/notes_ag/veg-TMV-ToMV
- **Publisher/type and page heading:** Utah State University Extension, Plant Health / Integrated Pest Management; “Tobacco Mosaic Virus & Tomato Mosaic Virus” (extension note).
- **Access/review:** Exact URL returned page text over HTTPS on 2026-10-24. “Biology” and “Symptoms” sections were reviewed.
- **Claims checked/evidence:** “Biology” says TMV and ToMV are tobamoviruses and describes spread by seed, artificial grafting, human handling, and tobacco products. “Symptoms” gives a combined range for the two viruses, including abnormal fruit, fruit lesions, reduced fruit size, distorted growing points, abnormal leaf color/patterns, distorted stems, and plant distortion/dwarfing.
- **Assessment:** The genus and combined symptom descriptions are supported as statements about the two-virus discussion. The page does not uniquely attribute the shared symptoms or every transmission route to ToMV alone. It does not specifically say “contaminated tools”; the JSON was narrowed to “human handling.”
- **Final audit status:** **Partially supported.**
- **Limitations and next action:** Keep the combined-virus attribution explicit. If ToMV-specific symptoms or transmission are needed, find a suitable authoritative source distinguishing ToMV from TMV. This page does not confirm ToMV in an individual image.

## Data and safety decisions

- No JSON source has been marked `verified`; `verified_on` and `verification_method` remain empty for every source. The three unauthenticated/inaccessible cases and the remaining limitations do not support complete source verification.
- The spider-mite background now reflects the retrieved page’s genus-level *Tetranychus* wording, while explicitly excluding species-level or image-level confirmation.
- The ToMV background now says “human handling,” as the USU source states, and avoids the unsupported specificity “contaminated tools.”
- No source was added or replaced. All source IDs and exact URLs were retained.
- `management_guidance` remains empty for all ten classes. No pesticide or treatment instructions were imported. The healthy class remains without disease sources or disease claims.
- Nothing in this literature audit establishes severity, progression, lesion segmentation, field diagnosis, or a confirmed diagnosis for an individual PlantVillage image.

## Remaining manual-review checklist

1. Human reviewer: review the complete source pages and their cited references for TNAU bacterial spot, early blight, late blight, Septoria, spider mites, and tomato leaf curl; cross-check organism nomenclature.
2. Human reviewer: access the Minnesota Extension page or an official accessible copy and compare every leaf-mold claim.
3. Retry the TNAU late-blight and spider-mite pages in a browser after certificate validation succeeds; do not rely on content fetched with TLS verification disabled.
4. Review the UF/IFAS publication with its Florida scope in view and obtain local Tamil Nadu evidence before making local applicability claims.
5. Obtain authoritative, virus-specific evidence for TYLCV and ToMV if class-specific organism claims are required.
6. Only after those reviews, update the audit, verification metadata, and tests to reflect the exact claims actually verified. Keep recommendations withheld unless a separate review establishes sufficient, relevant, verified management evidence.
