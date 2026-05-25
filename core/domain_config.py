from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DomainConfig:
    name: str
    structural_tag: str
    functional_tags: List[str]
    key_registration_fixed: str
    key_registration_moving: str
    token_aliases: Dict[str, str] = field(default_factory=dict)
    structural_keywords: List[str] = field(default_factory=list)
    functional_keywords: List[str] = field(default_factory=list)
    vlm_system_prompt: str = ""
    report_structure: List[str] = field(default_factory=list)

    def normalize_token(self, token: str) -> str:
        t = token.strip()
        if not t:
            return t
        for suffix in ("_registered", "-registered", "_resampled", "-resampled"):
            if t.lower().endswith(suffix):
                t = t[: -len(suffix)]
                t = t.strip("_- ")
                break
        low = t.lower()
        if "from_step" in low or "series_path" in low:
            inferred = self.infer_token_from_hint(low)
            if inferred:
                return inferred
        return self.token_aliases.get(low, t)

    def infer_token_from_hint(self, low: str) -> Optional[str]:
        if any(k in low for k in self.structural_keywords):
            return self.structural_tag
        for tag in self.functional_tags:
            if tag.lower() in low:
                return tag
        return None

    def resolve_token(self, token: Any, mapping: Dict[str, str]) -> Optional[str]:
        if not isinstance(token, str):
            return None
        t = token.strip()
        if not t:
            return None
        if "/" in t or t.startswith("."):
            return t
        norm = self.normalize_token(t)
        return mapping.get(norm)

    def structural_path(self, mapping: Dict[str, str]) -> Optional[str]:
        return mapping.get(self.structural_tag)

    def looks_structural(self, text: str) -> bool:
        low = text.lower()
        return any(k in low for k in self.structural_keywords)

    def looks_functional(self, text: str) -> bool:
        low = text.lower()
        return any(k in low for k in self.functional_keywords)

    def should_swap_registration(self, fixed_name: str, moving_name: str) -> bool:
        return self.looks_functional(fixed_name) and self.looks_structural(moving_name)

    def registration_subdir(self, moving_path: str) -> Optional[str]:
        low = moving_path.lower()
        for tag in self.functional_tags:
            if tag.lower() in low:
                return f"registration/{tag}"
        return None

    def canonical_image_name(self, name: str, path: str) -> str:
        n = str(name or "").strip()
        pl = str(path or "").lower()
        nl = n.lower()
        if any(k in pl or k in nl for k in self.structural_keywords):
            return self.structural_tag
        for tag in self.functional_tags:
            if tag.lower() in pl or tag.lower() in nl:
                return tag
        return n or Path(path).stem


@dataclass(frozen=True)
class ProstateDomainConfig(DomainConfig):
    def __init__(self) -> None:
        super().__init__(
            name="prostate",
            structural_tag="T2w",
            functional_tags=["ADC", "DWI"],
            key_registration_fixed="T2w",
            key_registration_moving="ADC",
            token_aliases={
                "t2w": "T2w",
                "t2": "T2w",
                "reference": "T2w",
                "fixed": "T2w",
                "dwi": "DWI",
                "adc": "ADC",
                "dwi_bval": "DWI_BVAL",
                "bval": "DWI_BVAL",
            },
            structural_keywords=["t2w", "t2"],
            functional_keywords=["low_b", "high_b", "dwi", "diffusion", "adc"],
            vlm_system_prompt=(
                "You are a radiology-style writer for prostate mpMRI engineering demos.\n"
                "Write a clear report in the EXACT Markdown structure requested (title + bold section headers + bullets).\n"
                "The VERY FIRST line must be exactly: '# MRI PROSTATE RADIOLOGY REPORT'.\n"
                "Always assign PI-RADS v2.1 scores (1-5) for each lesion and an overall score; do not say 'PI-RADS not generated'.\n"
                "PI-RADS guidance (simplified for this demo):\n"
                "- Peripheral zone (PZ, DWI-dominant): 1 normal DWI/ADC; 2 diffuse abnormal without focal lesion; "
                "3 focal DWI/ADC abnormal without typical T2; 4 marked low ADC/high b, <1.5 cm; "
                "5 same but >=1.5 cm or obvious extraprostatic extension.\n"
                "- Transition zone (TZ, T2-dominant): 1 typical BPH; 2 well-defined encapsulated nodule; "
                "3 atypical/intermediate; 4 ill-defined homogeneous low T2, <1.5 cm; 5 same but >=1.5 cm.\n"
                "Hard rules:\n"
                "- Ground every numeric statement in the provided evidence_bundle previews.\n"
                "- Do NOT hallucinate lesions. If evidence is insufficient, say 'indeterminate' and list limitations.\n"
                "- If you cite image-based observations, reference the montage tile coordinates using the provided tiles.json indices.\n"
                "- If candidates.json shows 0 candidates, state that explicitly in the **Quality/Limitations** bullet.\n"
                "- Output Markdown only (no JSON).\n"
            ),
            report_structure=[
                "# MRI PROSTATE RADIOLOGY REPORT",
                "**CLINICAL INDICATION:**",
                "**COMPARISON:**",
                "**TECHNICAL FACTORS:**",
                "**IMAGING MEDICATION:**",
                "---",
                "**FINDINGS:**",
                "**General Prostate:**",
                "**Transitional Zone (TZ):**",
                "**Peripheral Zone (PZ):**",
                "**Focal Lesion Evaluation (PI-RADS v2):**",
                "**Extra-Prostatic Findings:**",
                "---",
                "**IMPRESSION:**",
                "*Standardized reporting guidelines follow recommendations by ACR-ESUR PIRADS v2.*",
            ],
        )

    def registration_subdir(self, moving_path: str) -> Optional[str]:
        low = moving_path.lower()
        if "adc" in low:
            return "registration/ADC"
        if "dwi" in low or "trace" in low or "diff" in low:
            return "registration/DWI"
        return None

    def should_swap_registration(self, fixed_name: str, moving_name: str) -> bool:
        return "diffusion" in fixed_name.lower() and "t2" in moving_name.lower()

    def canonical_image_name(self, name: str, path: str) -> str:
        n = str(name or "").strip()
        pl = str(path or "").lower()
        nl = n.lower()
        if "t2w_input" in pl or "t2w_input" in nl:
            return "T2w_t2w_input"
        if "t2" in pl or "t2" in nl:
            return "T2w"
        if "adc" in pl or "adc" in nl:
            return "ADC"
        if "high_b" in pl or "high-b" in pl or "b1400" in pl:
            return "high_b"
        if "low_b" in pl or "low-b" in pl or "b50" in pl:
            return "low_b"
        if "mid_b" in pl or "mid-b" in pl or "b800" in pl:
            return "mid_b"
        if "dwi" in pl or "dwi" in nl:
            return "DWI"
        return n or Path(path).stem


@dataclass(frozen=True)
class BrainDomainConfig(DomainConfig):
    def __init__(self) -> None:
        super().__init__(
            name="brain",
            structural_tag="T1c",
            functional_tags=["FLAIR"],
            key_registration_fixed="T1c",
            key_registration_moving="FLAIR",
            token_aliases={
                "t1c": "T1c",
                "t1ce": "T1c",
                "t1gd": "T1c",
                "t1": "T1c",
                "reference": "T1c",
                "fixed": "T1c",
                "flair": "FLAIR",
                "t2": "T2",
            },
            structural_keywords=["t1c", "t1ce", "t1gd", "t1"],
            functional_keywords=["flair"],
            vlm_system_prompt=(
                "You are a neuroradiology assistant. Write a clear free-text report for a Brain Tumor MRI case. "
                "Ground findings in the provided evidence. Do NOT hallucinate."
            ),
            report_structure=[
                "Clinical Indication (Brain Tumor)",
                "Technique",
                "Findings (Tumor location, composition, mass effect)",
                "Impression",
            ],
        )

    def structural_path(self, mapping: Dict[str, str]) -> Optional[str]:
        return mapping.get("T1c") or mapping.get("T1")

    def canonical_image_name(self, name: str, path: str) -> str:
        n = str(name or "").strip()
        pl = str(path or "").lower()
        nl = n.lower()
        if "t1c" in pl or "t1ce" in pl or "t1gd" in pl:
            return "T1c"
        if "t1" in pl:
            return "T1"
        if "t1c" in nl or "t1ce" in nl or "t1gd" in nl:
            return "T1c"
        if "t1" in nl:
            return "T1"
        if "flair" in pl or "flair" in nl:
            return "FLAIR"
        if "t2" in pl or "t2" in nl:
            return "T2"
        return n or Path(path).stem


@dataclass(frozen=True)
class CardiacDomainConfig(DomainConfig):
    def __init__(self) -> None:
        super().__init__(
            name="cardiac",
            structural_tag="CINE",
            functional_tags=[],
            key_registration_fixed="CINE",
            key_registration_moving="CINE",
            token_aliases={
                "cine": "CINE",
                "bssfp": "CINE",
                "ssfp": "CINE",
                "sax": "CINE",
                "shortaxis": "CINE",
                "short_axis": "CINE",
                "reference": "CINE",
                "fixed": "CINE",
            },
            structural_keywords=["cine", "bssfp", "ssfp", "sax", "shortaxis", "short_axis"],
            functional_keywords=[],
            vlm_system_prompt=(
                "You are a cardiac MRI assistant for engineering demos. "
                "Ground findings in provided evidence and do not hallucinate."
            ),
            report_structure=[
                "Clinical Indication (Cardiac Cine MRI)",
                "Technique",
                "Findings (LV, RV, myocardium, disease class)",
                "Impression",
            ],
        )

    def registration_subdir(self, moving_path: str) -> Optional[str]:
        return None

    def should_swap_registration(self, fixed_name: str, moving_name: str) -> bool:
        return False

    def canonical_image_name(self, name: str, path: str) -> str:
        n = str(name or "").strip()
        pl = str(path or "").lower()
        nl = n.lower()
        if "cine" in pl or "bssfp" in pl or "ssfp" in pl or "cine" in nl:
            return "CINE"
        if "molli" in pl or "molli" in nl:
            return "T1_MOLLI"
        if "t1" in pl or "t1" in nl:
            return "T1"
        return n or Path(path).stem


def get_domain_config(name: str) -> DomainConfig:
    norm = (name or "").strip().lower()
    if norm == "brain":
        return BrainDomainConfig()
    if norm == "cardiac":
        return CardiacDomainConfig()
    return ProstateDomainConfig()
