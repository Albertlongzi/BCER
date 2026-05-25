from __future__ import annotations

from commands.registry import ToolRegistry

from .alignment_qc import build_tool as build_alignment_qc_tool
from .bm3d_denoising import build_tool as build_bm3d_denoise_tool
from .brain_glioma_grade_classification import build_tool as build_brain_glioma_grade_classification_tool
from .brain_tumor_segmentation import build_tool as build_brats_mri_segmentation_tool
from .cardiac_cine_classification import build_tool as build_cardiac_cine_classification_tool
from .cardiac_cine_segmentation import build_tool as build_cardiac_cine_segmentation_tool
from .compare_nifti_slices import build_tool as build_compare_nifti_tool
from .dicom_ingest import build_tools as build_dicom_tools
from .generate_qa_snapshot import build_tool as build_qa_snapshot_tool
from .materialize_registration import build_tool as build_materialize_registration_tool
from .prostate_distortion_correction import build_tool as build_prostate_distortion_tool
from .prostate_lesion_candidates import build_tool as build_lesion_candidates_tool
from .prostate_segmentation import build_tool as build_segmentation_tool
from .rag_search import build_tool as build_rag_search_tool
from .reconstruct_grappa import build_tool as build_grappa_tool
from .registration import build_tool as build_registration_tool
from .report_generation import build_tool as build_report_tool
from .resample_image import build_tool as build_resample_image_tool
from .roi_features import build_tool as build_feature_tool
from .sandbox_exec import build_tool as build_sandbox_tool
from .vlm_evidence import build_tool as build_package_vlm_evidence_tool


def build_registry(*, include_experimental: bool = True) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in build_dicom_tools():
        reg.register(tool)
    reg.register(build_registration_tool())
    reg.register(build_alignment_qc_tool())
    reg.register(build_materialize_registration_tool())
    reg.register(build_segmentation_tool())
    reg.register(build_brats_mri_segmentation_tool())
    reg.register(build_brain_glioma_grade_classification_tool())
    reg.register(build_cardiac_cine_segmentation_tool())
    reg.register(build_cardiac_cine_classification_tool())
    reg.register(build_feature_tool())
    reg.register(build_lesion_candidates_tool())
    reg.register(build_prostate_distortion_tool())
    reg.register(build_package_vlm_evidence_tool())
    reg.register(build_report_tool())
    reg.register(build_bm3d_denoise_tool())
    reg.register(build_resample_image_tool())
    reg.register(build_grappa_tool())
    reg.register(build_compare_nifti_tool())
    reg.register(build_qa_snapshot_tool())
    if include_experimental:
        reg.register(build_rag_search_tool())
        reg.register(build_sandbox_tool())
    return reg
