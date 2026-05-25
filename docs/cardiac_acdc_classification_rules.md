# Cardiac Cine Classification Rules (ACDC-Oriented)

This project uses rule-based cardiac cine classification in:
- `tools/cardiac_cine_classification.py`

Target classes:
- `NOR`
- `MINF`
- `DCM`
- `HCM`
- `RV`
- `UNCLASSIFIED` (borderline/insufficient)

## Core Physiological Signals
Computed from segmentation labels (`1=RV`, `2=MYO`, `3=LV`):
- LV/RV EDV, ESV, EF
- LV mass (myocardial volume at ED x 1.05 g/mL)
- Max myocardium thickness (ED frame)
- Local contraction proxy (slice-level myocardium area change ED->ES)

Optional from `Info.cfg`:
- ED/ES frame index
- Height/Weight for BSA-indexed volumes and mass
- Ground-truth group (`Group`)

## Ambiguity Rules Incorporated
1. HCM vs MINF:
- HCM requires LV EF > 55%.
- If LV EF < 40% and local myocardial thickness increase exists, classify as `MINF`.

2. MINF rule for dilated LV + regional abnormality:
- If LVEDV is high, LVEF is low, and only several segments are abnormally contracting, classify as `MINF`.

3. DCM with biventricular dilation:
- If both LV and RV are dilated (with or without RV dysfunction), classify as `DCM`.

4. Borderline RV function:
- RV EF > 45% is treated as normal.
- RV EF in `[40%, 45%)` is borderline and should not force `RV` class.

## Practical Fallbacks
- If high LVEDV + low LVEF and no MINF-specific pattern, classify as `DCM`.
- If none of the above rules are decisive, output `UNCLASSIFIED` and mark `needs_vlm_review=true`.

## Notes
- The local contraction criterion is currently a proxy from slice-level myocardium area change, not full AHA segment strain analysis.
- For ambiguous cases, VLM-assisted qualitative review is recommended.
