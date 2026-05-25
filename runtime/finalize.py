from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.domain_config import DomainConfig


@dataclass(frozen=True)
class FinalizeResult:
    status: str
    evidence_bundle_path: Optional[str]
    final_report_txt_path: str
    final_report_json_path: str


def _write_llm_debug(artifacts_dir: Path, tag: str, *, messages_obj: List[Dict[str, Any]], raw_text: str) -> None:
    dbg = artifacts_dir / "llm_debug"
    dbg.mkdir(parents=True, exist_ok=True)
    (dbg / f"{tag}_messages.json").write_text(json.dumps(messages_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (dbg / f"{tag}_raw.txt").write_text((raw_text or "") + "\n", encoding="utf-8")


def _encode_png_data_url(path: str, *, max_bytes: int = 2_500_000) -> Optional[str]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None
    try:
        b = p.read_bytes()
    except Exception:
        return None
    if len(b) > max_bytes:
        return None
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def _pick_overlay_paths(evidence_bundle: Dict[str, Any], *, max_images: int = 4) -> List[str]:
    overlays: List[Any] = []
    try:
        feat = evidence_bundle.get("features") if isinstance(evidence_bundle, dict) else {}
        overlays = (feat.get("overlay_pngs") if isinstance(feat, dict) else []) or []
    except Exception:
        overlays = []

    overlays = [
        o
        for o in overlays
        if isinstance(o, dict)
        and o.get("path")
        and isinstance(o.get("kind"), str)
        and str(o.get("kind")).startswith(("roi_crop_montage_", "full_frame_montage_"))
    ]

    def _bucket(path: str) -> str:
        s = str(path).lower()
        if "t2w" in s:
            return "t2w"
        if "adc" in s:
            return "adc"
        if "high_b" in s:
            return "dwi_high_b"
        if "low_b" in s:
            return "dwi_low_b"
        return "other"

    picked: Dict[str, str] = {}
    for o in overlays:
        p = str(o.get("path"))
        b = _bucket(p)
        if b not in picked:
            picked[b] = p
    return [picked[b] for b in ("t2w", "adc", "dwi_high_b", "dwi_low_b") if b in picked][:max_images]


def _load_tiles_index_from_overlays(evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    overlays: List[Any] = []
    try:
        feat = evidence_bundle.get("features") if isinstance(evidence_bundle, dict) else {}
        overlays = (feat.get("overlay_pngs") if isinstance(feat, dict) else []) or []
    except Exception:
        overlays = []

    tiles_index: Dict[str, Any] = {}
    try:
        for o in overlays:
            if not isinstance(o, dict):
                continue
            p = str(o.get("path", ""))
            seq = str(o.get("sequence", ""))
            if not p or not seq:
                continue
            tiles_p = Path(p).with_name(Path(p).name.replace(".png", "_tiles.json"))
            if tiles_p.exists():
                tiles_index[seq] = json.loads(tiles_p.read_text(encoding="utf-8"))
    except Exception:
        tiles_index = {}
    return tiles_index


def _load_candidates(evidence_bundle: Dict[str, Any], *, artifacts_dir: Path) -> Tuple[List[Any], Optional[float], Optional[str]]:
    lesion_candidates: Dict[str, Any] = {}
    candidates_list: List[Any] = []
    candidates_threshold: Optional[float] = None
    cand_path_used: Optional[str] = None
    try:
        lesion_block = evidence_bundle.get("lesion") if isinstance(evidence_bundle, dict) else None
        cand_path = None
        if isinstance(lesion_block, dict):
            cand_path = lesion_block.get("candidates_path")
        if not cand_path:
            cand_path = str((artifacts_dir / "lesion" / "candidates.json").resolve())
        cand_p = Path(str(cand_path)).expanduser().resolve()
        cand_path_used = str(cand_p)
        if cand_p.exists():
            lesion_candidates = json.loads(cand_p.read_text(encoding="utf-8"))
    except Exception:
        lesion_candidates = {}
    if isinstance(lesion_candidates, dict):
        candidates_list = lesion_candidates.get("candidates") or []
        candidates_threshold = lesion_candidates.get("threshold")
        if not isinstance(candidates_list, list):
            candidates_list = []
    return candidates_list, candidates_threshold, cand_path_used


def _postprocess_report_text(
    txt: str,
    *,
    candidates_list: List[Any],
    candidates_threshold: Optional[float],
    b_values_available: bool,
    domain_name: Optional[str] = None,
) -> str:
    cleaned = (txt or "").strip()
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()

    is_prostate = (domain_name or "").strip().lower() == "prostate"
    if is_prostate:
        # Preserve a markdown title line if present; otherwise trim to the first recognizable header.
        title_re = re.search(r"(?mi)^#\s*MRI PROSTATE RADIOLOGY REPORT\s*$", cleaned)
        if title_re:
            cleaned = cleaned[title_re.start() :].lstrip()
        else:
            m_hdr = re.search(r"(?mi)^\*\*CLINICAL INDICATION:\*\*\s*$", cleaned)
            if m_hdr:
                cleaned = cleaned[m_hdr.start() :].lstrip()
            else:
                up = cleaned.upper()
                idx = up.find("CLINICAL INDICATION:")
                if idx >= 0:
                    cleaned = cleaned[idx:].lstrip()

        # Normalize inline clinical indication headers to standalone markdown lines.
        lines = cleaned.splitlines()
        for i, line in enumerate(list(lines)):
            m_inline = re.match(r"(?i)^\*\*CLINICAL INDICATION:\*\*\s*(.*)\s*$", line)
            if m_inline:
                trailing = (m_inline.group(1) or "").strip()
                if trailing and re.search(r"(?i)\b(use|ground|hallucinat|output rules|required section)\b", trailing):
                    trailing = ""
                lines[i] = "**CLINICAL INDICATION:**"
                if trailing:
                    lines.insert(i + 1, trailing)
                cleaned = "\n".join(lines).lstrip()
                break
    else:
        m_hdr = re.search(r"(?mi)^CLINICAL INDICATION:\s*$", cleaned)
        if m_hdr:
            cleaned = cleaned[m_hdr.start() :].lstrip()
        else:
            up = cleaned.upper()
            idx = up.find("CLINICAL INDICATION:")
            if idx >= 0:
                cleaned = cleaned[idx:].lstrip()

        # Enforce that the report starts with a standalone header line "CLINICAL INDICATION:".
        # If the model puts content on the same line (or emits prompt-echo garbage), normalize it.
        lines = cleaned.splitlines()
        if lines:
            m_inline = re.match(r"(?i)^CLINICAL INDICATION:\s*(.*)\s*$", lines[0])
            if m_inline:
                trailing = (m_inline.group(1) or "").strip()
                # Heuristic: drop common prompt-echo fragments.
                if trailing and re.search(r"(?i)\b(use|ground|hallucinat|output rules|required section)\b", trailing):
                    trailing = ""
                new_lines = ["CLINICAL INDICATION:"]
                if trailing:
                    new_lines.append(trailing)
                new_lines.extend(lines[1:])
                cleaned = "\n".join(new_lines).lstrip()

    if not b_values_available and re.search(r"(?i)\bb\s*=\s*\d+", cleaned):
        cleaned = re.sub(r"(?i)\(\s*b\s*=\s*\d+[^\)]*\)", "", cleaned)
        cleaned = re.sub(r"(?i)\bb\s*=\s*\d+\s*(s/mm\^2|s/mm²)?", "b=unknown", cleaned)

    if isinstance(candidates_list, list) and len(candidates_list) == 0:
        if not re.search(r"(?i)0 candidates", cleaned):
            statement = (
                "Tool-generated lesion candidates: 0 candidates"
                + (f" at threshold {candidates_threshold}" if candidates_threshold is not None else "")
                + " (from candidates.json)."
            )
            if is_prostate:
                lines = cleaned.splitlines()
                # Try to insert under General Prostate -> Quality/Limitations.
                gp_idx = None
                for i, line in enumerate(lines):
                    if line.strip().lower() == "**general prostate:**":
                        gp_idx = i
                        break
                if gp_idx is not None:
                    end_idx = len(lines)
                    for j in range(gp_idx + 1, len(lines)):
                        cand = lines[j].strip()
                        if not cand:
                            continue
                        if cand.startswith("**") and cand.endswith(":"):
                            end_idx = j
                            break
                        if cand.startswith("---") or cand.startswith("#"):
                            end_idx = j
                            break
                    q_idx = None
                    for j in range(gp_idx + 1, end_idx):
                        if "quality/limitations" in lines[j].lower():
                            q_idx = j
                            break
                    if q_idx is not None:
                        if statement not in lines[q_idx]:
                            lines[q_idx] = lines[q_idx].rstrip() + f" {statement}"
                    else:
                        insert_at = end_idx
                        lines.insert(insert_at, f"* **Quality/Limitations:** {statement}")
                else:
                    lines.append("")
                    lines.append(f"* **Quality/Limitations:** {statement}")
                cleaned = "\n".join(lines).lstrip()
            else:
                m_lim = re.search(r"(?mi)^LIMITATIONS:\s*$", cleaned)
                if m_lim:
                    insert_pos = m_lim.end()
                    cleaned = cleaned[:insert_pos] + "\n" + statement + "\n" + cleaned[insert_pos:].lstrip()
                else:
                    cleaned = cleaned.rstrip() + "\n\nLIMITATIONS:\n" + statement + "\n"
    return cleaned


def _build_section_headers(report_structure: List[str]) -> List[str]:
    lines: List[str] = []
    for section in report_structure:
        if not isinstance(section, str):
            continue
        text = section.strip()
        if not text:
            continue
        if text.startswith(("#", "**", "*", "---", ">")):
            lines.append(text)
            continue
        base = text
        hint = ""
        if "(" in text and text.endswith(")"):
            base, hint = text[:-1].split("(", 1)
            base = base.strip()
            hint = hint.strip()
        header = base.upper()
        if not header.endswith(":"):
            header += ":"
        if hint:
            header = f"{header} {hint}".strip()
        lines.append(header)
    return lines


def finalize_free_text_report(
    *,
    llm: Any,
    llm_mode: str,
    artifacts_dir: Path,
    run_dir: Path,
    case_id: str,
    goal: str,
    domain_config: Optional[DomainConfig] = None,
) -> FinalizeResult:
    """
    Shared finalize-with-llm implementation:
    - reads packaged evidence bundle
    - attaches a small number of montage PNGs (data URLs) if size permits
    - enforces strict post-processing constraints (TECHNIQUE start, candidates.json, no b-value hallucination)
    - writes artifacts/final/final_report.txt and artifacts/final/final_report.json
    """
    final_dir = artifacts_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = artifacts_dir / "vlm" / "vlm_evidence_bundle.json"
    if not evidence_path.exists():
        found = sorted(artifacts_dir.glob("**/vlm_evidence_bundle.json"))
        if found:
            evidence_path = found[0]

    evidence_bundle: Dict[str, Any] = {}
    try:
        if evidence_path.exists():
            evidence_bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        evidence_bundle = {}

    tiles_index = _load_tiles_index_from_overlays(evidence_bundle)
    candidates_list, candidates_threshold, _cand_path_used = _load_candidates(evidence_bundle, artifacts_dir=artifacts_dir)

    dicom_summary = evidence_bundle.get("dicom_summary") if isinstance(evidence_bundle, dict) else {}
    b_values_available = bool(dicom_summary.get("diffusion_b_values")) if isinstance(dicom_summary, dict) else False

    default_system_prompt = (
        "You are a radiology-style prostate mpMRI assistant.\n"
        "Write a clinically-useful FREE-TEXT report in a PI-RADS v2.1-style organization (but do not quote the standard).\n"
        "CRITICAL output rules:\n"
        "- Output ONLY the final report text. NO preamble. NO planning. NO chain-of-thought.\n"
        "- The VERY FIRST characters of your output MUST be exactly: 'CLINICAL INDICATION:'\n"
        "- Use the exact section headers below (all-caps), in this order.\n"
        "- Ground every numeric statement in evidence_bundle.features_preview_rows / slice_summary_preview / lesion_geometry.\n"
        "- Do NOT hallucinate lesions.\n"
        "- Do NOT state specific DWI b-values unless evidence_bundle explicitly provides them.\n"
        "- candidates.json is authoritative. If candidates=[] you MUST state '0 candidates at threshold' and MUST NOT invent candidates.\n"
        "- If you describe any lesion or suspicious focus, you MUST cite tiles (r<row> c<col> z<z>) and use lesion_geometry when available.\n"
        "- You MAY describe an image-based *suspicious focus* ONLY if you cite tiles AND the focus shows a cross-sequence pattern "
        "(high_b relatively high + ADC relatively low, and/or corresponding T2 abnormality). Otherwise say lesion assessment indeterminate.\n"
        "- You MUST look at the attached montage images and include QC/artifact notes with tile citations in FINDINGS or LIMITATIONS.\n"
        "- For tile citations, use the montage tile labels (r<row> c<col> z<z>) and/or tiles_index provided.\n"
        "- If clinical indication, comparison, or medication details are missing, explicitly write 'Not provided'.\n"
        "Required section headers:\n"
        "CLINICAL INDICATION:\n"
        "COMPARISON:\n"
        "TECHNICAL FACTORS:\n"
        "IMAGING MEDICATION:\n"
        "FINDINGS:\n"
        "IMPRESSION:\n"
        "LIMITATIONS:\n"
    )
    if domain_config is None:
        system_prompt = default_system_prompt
    else:
        base_rules = [
            "Output ONLY the final report text. NO preamble. NO planning. NO chain-of-thought.",
            "Use the exact section headers below, in this order.",
            "Ground every numeric statement in evidence_bundle.features_preview_rows / slice_summary_preview / lesion_geometry.",
            "Do NOT hallucinate lesions.",
            "If you describe any lesion or suspicious focus, you MUST cite tiles (r<row> c<col> z<z>) and use lesion_geometry when available.",
            "You MUST look at the attached montage images and include QC/artifact notes with tile citations in FINDINGS or LIMITATIONS.",
            "For tile citations, use the montage tile labels (r<row> c<col> z<z>) and/or tiles_index provided.",
            "If clinical indication, comparison, or medication details are missing, explicitly write 'Not provided'.",
        ]
        prostate_rules = [
            "The VERY FIRST line of your output MUST be exactly: '# MRI PROSTATE RADIOLOGY REPORT'",
            "Use Markdown formatting with bold section headers and bullets exactly as requested.",
            "Always provide a PI-RADS score (1-5) for each lesion and an overall score, even if indeterminate.",
            "Do not add extra sections beyond the required headers; place limitations in the Quality/Limitations bullet.",
            "Do NOT state specific DWI b-values unless evidence_bundle explicitly provides them.",
            "candidates.json is authoritative. If candidates=[] you MUST state '0 candidates at threshold' and MUST NOT invent candidates.",
            (
                "You MAY describe an image-based *suspicious focus* ONLY if you cite tiles AND the focus shows a cross-sequence pattern "
                "(high_b relatively high + ADC relatively low, and/or corresponding T2 abnormality). Otherwise say lesion assessment indeterminate."
            ),
        ]
        other_rules = [
            "The VERY FIRST characters of your output MUST be exactly: 'CLINICAL INDICATION:'",
        ]
        rules = base_rules + (prostate_rules if domain_config.name == "prostate" else other_rules)
        section_lines = _build_section_headers(domain_config.report_structure)
        if not section_lines:
            section_lines = _build_section_headers(
                [
                    "CLINICAL INDICATION",
                    "COMPARISON",
                    "TECHNICAL FACTORS",
                    "IMAGING MEDICATION",
                    "FINDINGS",
                    "IMPRESSION",
                    "LIMITATIONS",
                ]
            )
        system_prompt = (
            f"{domain_config.vlm_system_prompt}\n"
            "CRITICAL output rules:\n"
            + "\n".join(f"- {rule}" for rule in rules)
            + "\nRequired section headers:\n"
            + "\n".join(section_lines)
            + "\n"
        )
    user_payload = {
        "goal": goal,
        "case_id": case_id,
        "evidence_bundle_path": str(evidence_path) if evidence_path.exists() else None,
        "evidence_bundle": evidence_bundle,
        "dicom_summary": dicom_summary,
        "lesion_geometry": evidence_bundle.get("lesion_geometry") if isinstance(evidence_bundle, dict) else None,
        "tiles_index": tiles_index,
        "lesion_candidates": {"threshold": candidates_threshold, "count": int(len(candidates_list)), "candidates": candidates_list},
        "b_values_available": b_values_available,
        "image_qc_tasks": [
            "ROI mask coverage: identify any tiles where mask appears clearly too large/too small/fragmented; cite tile(s).",
            "Registration alignment: compare T2w vs DWI/ADC appearances within the ROI crop; if systematic offset is visible, cite tile(s); otherwise say indeterminate.",
            "Any obvious artifacts (motion/ghosting/signal dropout) impacting interpretability; cite tile(s) if visible.",
        ],
        "lesion_screen_tasks": [
            "If detect_lesion_candidates outputs are present in evidence_bundle, review boxed candidates first and say whether each looks plausible, citing tiles.",
            "If no candidates are provided by tools, still scan montages; only report suspicious focus if cross-sequence pattern is present and you can cite tiles; otherwise indeterminate.",
        ],
    }

    # Try with 4 images -> 2 -> 0 to avoid context blowups from base64.
    attempts = [4, 2, 0]
    last_err: Optional[str] = None
    txt = ""
    used_images = 0
    for n in attempts:
        chosen_paths = _pick_overlay_paths(evidence_bundle, max_images=n)
        img_blocks: List[Dict[str, Any]] = []
        for p in chosen_paths:
            du = _encode_png_data_url(p)
            if du:
                img_blocks.append({"type": "image_url", "image_url": {"url": du}})
        used_images = len(img_blocks)
        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}] + img_blocks},
        ]

        if llm_mode != "server":
            txt = ""
            break
        try:
            txt = llm.generate(final_messages)
            last_err = None
            break
        except Exception as e:
            last_err = repr(e)
            _write_llm_debug(artifacts_dir, "final_free_text_error", messages_obj=final_messages, raw_text=last_err)
            # Retry on context length errors by reducing images
            if "maximum context length" in last_err.lower() or "400 bad request" in last_err.lower():
                continue
            break

    final_txt = final_dir / "final_report.txt"
    if txt.strip():
        cleaned = _postprocess_report_text(
            txt,
            candidates_list=candidates_list,
            candidates_threshold=candidates_threshold,
            b_values_available=b_values_available,
            domain_name=(domain_config.name if domain_config else None),
        )
        final_txt.write_text(cleaned + "\n", encoding="utf-8")
        status = "completed"
    else:
        final_txt.write_text(
            "VLM finalization unavailable; evidence bundle has been packaged for downstream reporting.\n",
            encoding="utf-8",
        )
        status = "completed_without_llm"

    final_json = final_dir / "final_report.json"
    payload = {
        "final_report": {
            "status": status,
            "run_dir": str(run_dir),
            "evidence_bundle_path": str(evidence_path) if evidence_path.exists() else None,
            "final_report_txt_path": str(final_txt),
            "meta": {"timestamp": datetime.utcnow().isoformat() + "Z", "used_images": used_images, "last_error": last_err},
        }
    }
    final_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return FinalizeResult(
        status=status,
        evidence_bundle_path=str(evidence_path) if evidence_path.exists() else None,
        final_report_txt_path=str(final_txt),
        final_report_json_path=str(final_json),
    )
