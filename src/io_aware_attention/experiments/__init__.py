"""Experiment runners for Trainium scaling studies."""

from io_aware_attention.experiments.kernel_study import (
    KernelStudyConfig,
    load_kernel_study_config,
    run_kernel_study,
)
from io_aware_attention.experiments.phase_study import (
    PhaseStudyConfig,
    load_phase_study_config,
    run_phase_study,
)

__all__ = [
    "KernelStudyConfig",
    "PhaseStudyConfig",
    "load_kernel_study_config",
    "load_phase_study_config",
    "run_kernel_study",
    "run_phase_study",
]
