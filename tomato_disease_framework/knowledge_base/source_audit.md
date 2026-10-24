# Phase 10A candidate-source first-pass audit

**Review date:** 2026-10-24
**Audit type:** Online preliminary content review of source text returned by the available page-fetch tool. This is not a complete independent manual verification.
**Verification policy:** All nine source records remain `unverified` in `tomato_diseases.json`; `verified_on` and `verification_method` remain empty. Page availability, publisher identity, and schema-valid provenance do not by themselves establish the truth or applicability of a claim. No management guidance or recommendation is authorized.

## Review summary

- Retrieved and reviewed page text for seven exact JSON URLs: five TNAU pages (bacterial spot, early blight, late blight, Septoria leaf spot, and tomato leaf curl), the UF/IFAS target-spot publication, and the Utah State University Extension TMV/ToMV page. The early-blight page was returned as raw HTML and its relevant section was read.
- Retrieved the spider-mite page text only at the same path on `https://agritech.tnau.ac.in` without `www`. The exact JSON URL with `www` could not be fetched by the available clients because of TLS/DNS failures. Treat this as a host-variant content check, not successful access to the exact URL.
- The University of Minnesota Extension leaf-mold URL returned HTTP 403. Its content and claims were not checked.
- No source was changed to `verified`. These fetches support preliminary comparison of selected visible text with the JSON statements, but do not complete full-page, date, attribution, or independent scientific review.

## Per-source review records

### 1. Bacterial spot — `tnau_tomato_bacterial_leaf_spot`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_11.html
- **Publisher/title:** The retrieved page is on the TNAU Agritech Portal and labels the section “Bacterial leaf spot”.
- **Access/review status:** Exact URL returned page text; relevant text reviewed in this preliminary pass.
- **Evidence location:** “Bacterial leaf spot”, “Field Diagnostic Symptoms”, “Causal organism”, and “Survival and Mode of Spread”.
- **Claims checked:** The page identifies *Xanthomonas axonopodis* pv. *vesicatoria*; lists small brown water-soaked circular leaf spots with yellow halos, water-soaked green-fruit spots that become sunken/scabby, infected debris survival, and infected debris/seed primary spread.
- **Assessment:** Background and symptom statements are **supported in substance** by the retrieved section. JSON wording “Xanthomonas-associated” is less taxonomically specific than the page. The source reports fruit symptoms as well as leaf symptoms; this does not identify symptoms in any particular input image.
- **Geographic/taxonomic limitations:** TNAU source and recommendations are region-specific; the pathogen name follows this page's nomenclature and may require taxonomic cross-checking.
- **Next action:** A human reviewer should compare the complete page and cited underlying references before considering verification. Keep recommendation guidance empty.

### 2. Early blight — `tnau_tomato_early_blight`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html
- **Publisher/title:** TNAU Agritech Portal; retrieved HTML section title “Early Blight: *Alternaria solani*”.
- **Access/review status:** Exact URL returned raw HTML on retry; relevant HTML section was inspected.
- **Evidence location:** “Field Diagnostic Symptoms”, “Causal organism”, “Survival and Mode of Spread”.
- **Claims checked:** The section names *Alternaria solani*, brown spots with concentric bull’s-eye rings and yellow margins, brown concentric fruit rings, infected plant debris survival, infected seed primary spread, and conidia carried by rain/wind.
- **Assessment:** The record’s organism, lesion, fruit-ring, debris, rain, and wind claims are **supported in substance**. The record omits the page’s infected-seed spread route; this is an omission, not a contradiction. Avoid implying that concentric rings uniquely diagnose early blight; the source content does not establish image-level diagnosis.
- **Geographic/taxonomic limitations:** TNAU/Indian agricultural context; no evidence here establishes performance in another region or in a particular image.
- **Next action:** Human full-page review and independent scientific cross-check remain necessary; keep source unverified.

### 3. Late blight — `tnau_tomato_late_blight`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html
- **Publisher/title:** TNAU Agritech Portal; retrieved heading “Late blight: *Phytophthora infestans*”.
- **Access/review status:** Exact URL returned page text; relevant content reviewed in this preliminary pass.
- **Evidence location:** “Symptoms”, “Fruit”, “Causal organism”, and “Survival and Mode of Spread”.
- **Claims checked:** Water-soaked black lesions on leaves/stems, rapid necrosis, white sporulation, dark-brown fruit lesions and soft rot/disintegration; *P. infestans*; infected-debris survival/primary spread and wind secondary spread.
- **Assessment:** Record background and symptom claims are **supported in substance** by the retrieved section. “Spread associated with infected debris and wind” is a concise summary of different routes listed on the page.
- **Geographic/taxonomic limitations:** TNAU/Indian agricultural context; pathogen naming on the page is the basis for the claim, not laboratory confirmation of an image.
- **Next action:** Full human review and independent cross-check remain necessary; keep source unverified.

### 4. Leaf mold — `umn_tomato_leaf_mold`

- **Exact URL:** https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold
- **Publisher/title:** Source record lists University of Minnesota Extension, “Tomato leaf mold”; not independently confirmed from page content in this pass.
- **Access/review status:** Fetch returned HTTP 403. Page content was **not checked**.
- **Evidence location:** None available in this pass.
- **Claims checked:** None. The JSON's *Passalora fulva* synonym list, high-humidity/greenhouse association, upper/lower leaf symptom contrast, and withering claims remain unverified here.
- **Assessment:** **Not assessed** because content was inaccessible; do not treat the existing claims as supported by this audit.
- **Geographic/taxonomic limitations:** The JSON notes US greenhouse/high-tunnel context; this and the taxonomic synonyms require source-level review.
- **Next action:** Human reviewer should access the page in a browser or obtain an accessible official copy and check each named claim and the page's publication/update metadata.

### 5. Septoria leaf spot — `tnau_tomato_septoria_leaf_spot`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_4.html
- **Publisher/title:** TNAU Agritech Portal; retrieved heading “Septoria Leaf Spot: *Septoria lycopersici*”.
- **Access/review status:** Exact URL returned page text; relevant content reviewed.
- **Evidence location:** “Symptoms”, “Causal Organism”, and “Survival and Mode of Spread”.
- **Claims checked:** Small round-to-irregular spots with gray centers/dark margins, coalescence and blighting; *S. lycopersici*; infected plant debris; conidia spread by rain splash or wind.
- **Assessment:** Background and symptom claims are **supported in substance**. The JSON's description is a subset of the page's symptom list; the page also mentions complete defoliation and affected stems/flowers.
- **Geographic/taxonomic limitations:** TNAU/Indian context; lesions overlap with other conditions and are not diagnostic by themselves.
- **Next action:** Complete human/source review remains necessary; keep source unverified.

### 6. Spider mites — `tnau_tomato_red_spider_mite`

- **Exact JSON URL:** https://www.agritech.tnau.ac.in/crop_protection/tomato/tomato_7.html
- **Publisher/title:** The source record names TNAU and “Red spider mite: *Tetranychus* spp.” Retrieved content at the same path on the non-`www` TNAU host labels the page “Red spider mite”; it did not show a genus/species name in the returned excerpt.
- **Access/review status:** Exact `www` URL fetch failed TLS/DNS checks. Same-path `https://agritech.tnau.ac.in/crop_protection/tomato/tomato_7.html` returned content; this does not prove exact-URL availability or that both host variants are identical.
- **Evidence location:** Retrieved non-`www` page sections “Symptoms of damage” and “Identification of pest”; images were captioned “Infestation on tomato” and “Red spider mite”.
- **Claims checked:** Returned page text supports reddish-brown/bronzy leaves, webbing and drying in severe infestation, and effects on flower/fruit formation. The retrieved text does **not** support the record's *Tetranychus* genus assertion.
- **Assessment:** Symptom/damage statement is **supported in substance** by the alternate-host content. Taxonomic statement is **unsupported by the retrieved excerpt** and has been narrowed in the JSON: it now says the accessible page does not identify genus/species. The source does not establish *T. urticae* or any species in individual dataset images.
- **Geographic/taxonomic limitations:** TNAU source; “red spider mite” is not sufficient species identification. Dataset class name includes “Two-spotted spider mite”, but that label is not specimen-level confirmation.
- **Next action:** Check the exact `www` URL in a normal browser; inspect the full identification section or find an authoritative source that explicitly identifies the species before making a taxonomic statement.

### 7. Target spot — `uf_ifas_target_spot_tomato`

- **Exact URL:** https://ask.ifas.ufl.edu/publication/PP351
- **Publisher/title:** UF/IFAS Extension publication PP351; page content identifies target spot of tomato and Florida context.
- **Access/review status:** Exact URL returned article text; relevant sections reviewed in this preliminary pass.
- **Evidence location:** Opening disease description and symptom paragraphs, including leaf-lesion, fruit-lesion, and inner-canopy descriptions.
- **Claims checked:** Article identifies *Corynespora cassiicola* as causal agent, states pathogen occurrence in tropical/subtropical and greenhouse production, describes dark pinpoint lesions enlarging to light-brown/gray centers with dark concentric rings and sometimes yellowing, fruit flecks that become deeply pitted, and disease developing in the inner canopy.
- **Assessment:** Revised JSON background and symptom statements are **supported in substance** by the retrieved content. The article also emphasizes confusion with early bacterial spot, early blight, and gray leaf spot; avoid implying symptom specificity.
- **Geographic/taxonomic limitations:** Publication focuses on Florida. Its pathogen-range statement is not evidence of identical disease prevalence, epidemiology, or management in Tamil Nadu.
- **Next action:** Human full-publication review and local applicability review remain necessary; keep source unverified.

### 8. Tomato yellow leaf curl — `tnau_tomato_leaf_curl`

- **Exact URL:** https://agritech.tnau.ac.in/crop_protection/tomato_diseases_6.html
- **Publisher/title:** TNAU Agritech Portal; retrieved heading “Leaf curl: Tomato leaf curl virus”.
- **Access/review status:** Exact URL returned page text; relevant content reviewed in this preliminary pass.
- **Evidence location:** “Field Diagnostic Symptoms”, “Causal organism”, “Vector”, and “Survival and Mode of Spread”.
- **Claims checked:** The page names “Tomato leaf curl virus (TLCV)”, whitefly *Bemisia tabaci*, whitefly primary spread and movement of infected plants; lists severe stunting, downward rolling/crinkling, leathery/brittle older leaves, shortened internodes, and bushy appearance.
- **Assessment:** Existing symptom and vector/spread statements are **supported in substance**. The page does not establish the exact TYLCV species/strain-level identity implied by the PlantVillage class. The JSON's caveat that this is not molecular confirmation is appropriate; retain it.
- **Geographic/taxonomic limitations:** TNAU's “tomato leaf curl virus” label is broader than the dataset's TYLCV class; visual features cannot confirm virus identity.
- **Next action:** Obtain a molecularly specific authoritative TYLCV source if the record is to make claims specific to that virus; keep this TNAU source unverified.

### 9. Tomato mosaic virus — `usu_tobacco_tomato_mosaic_virus`

- **Exact JSON URL:** https://extension.usu.edu/planthealth/ipm/notes_ag/veg-TMV-ToMV
- **Publisher/title:** Retrieved page heading “Tobacco Mosaic Virus & Tomato Mosaic Virus”, Utah State University Extension.
- **Access/review status:** Exact URL returned page text; relevant content reviewed in this preliminary pass.
- **Evidence location:** “Hosts”, “Biology”, “Symptoms”, and “General Management”.
- **Claims checked:** Page says TMV and ToMV are tobamoviruses; discusses seed, handling, grafting, tobacco products, plant debris, weedy hosts/soil; its combined symptom description includes malformed fruit, lesions, reduced fruit size, distorted growing points, abnormal color/patterns, distorted stems, and plant distortion/dwarfing.
- **Assessment:** ToMV is explicitly named, so this is not merely an unrelated-virus source. However, biology and symptom sections group TMV and ToMV, so the JSON's ToMV-specific wording is **partially supported** rather than uniquely attributed to ToMV. “Light and dark green mosaic patterns” is not explicit in the retrieved text; the page says abnormal leaf color and patterns. The JSON wording was narrowed to that combined symptom language and makes the shared-source limitation explicit.
- **Geographic/taxonomic limitations:** USU/Utah context. The source combines TMV and ToMV and does not diagnose any particular PlantVillage image.
- **Next action:** The task description mentions a TNAU candidate, but the actual JSON points to USU; no TNAU URL is present in the current record. Keep this USU source unverified and consider a TNAU/ICAR/other authoritative source specifically distinguishing ToMV if that distinction is needed. Do not invent a replacement URL.

## Data decisions from this first pass

- All nine sources remain `unverified`; `verified_on` and `verification_method` are empty. The page fetches provide preliminary content evidence for seven records but do not constitute the complete independent manual verification required by the project policy.
- The spider-mite background wording was narrowed because retrieved content did not establish the genus *Tetranychus*.
- The target-spot background and symptom wording was aligned more closely with the retrieved UF/IFAS text and retains the Florida limitation.
- The tomato-mosaic-virus symptom wording was narrowed because the USU page groups TMV and ToMV symptoms and does not explicitly state the light/dark-green mosaic description.
- The other claim text was left unchanged in this first pass; unresolved or inaccessible sources are explicitly identified above.
- `publication_date` remains blank for all records. No pesticide instructions or management guidance were copied; every `management_guidance` array remains empty.
- The healthy class has no disease-source requirement, source, or disease claims.

## Research limitations and next steps

1. This is a preliminary text review of retrieved page content, not a completed human verification of the whole sources or an independent scientific review.
2. Manually inspect the Minnesota page, exact spider-mite URL, and TNAU early-blight page if a direct browser verification record is required; the TNAU early-blight text was obtained as raw HTML in this environment.
3. Independently review the complete seven accessible page contents, update dates, attribution, and exact claim wording before changing any source to `verified`.
4. Validate geographic applicability before using non-Tamil Nadu guidance.
5. No symptoms in an input image are confirmed by these literature statements. No severity, progression, lesion segmentation, field generalization, or treatment recommendation is established.
