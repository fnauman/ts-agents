"""Sub-agents for the deep agent architecture.

Each sub-agent is a specialist in a specific type of time series analysis:
- decomposition: Trend/seasonal decomposition method selection
- forecasting: Model selection and uncertainty quantification
- patterns: Motifs, anomalies, and regime detection
- classification: Time series classification algorithm selection
- turbulence: CFD/turbulence-specific analysis (spectral, coherence)
"""

from .decomposition import DECOMPOSITION_SUBAGENT
from .forecasting import FORECASTING_SUBAGENT
from .patterns import PATTERNS_SUBAGENT
from .classification import CLASSIFICATION_SUBAGENT
from .turbulence import TURBULENCE_SUBAGENT

__all__ = [
    "DECOMPOSITION_SUBAGENT",
    "FORECASTING_SUBAGENT",
    "PATTERNS_SUBAGENT",
    "CLASSIFICATION_SUBAGENT",
    "TURBULENCE_SUBAGENT",
]
