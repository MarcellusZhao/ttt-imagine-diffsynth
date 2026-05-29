from .aesthetic import AestheticFilter
from .base import FilterResult, VideoFilter
from .black_border import BlackBorderFilter
from .blur import BlurFilter
from .duration_resolution import DurationResolutionFilter
from .motion_stability import MotionStabilityFilter
from .nsfw import NSFWFilter
from .overexposure import OverexposureFilter
from .text_detection import TextCoverageFilter
from .watermark import WatermarkFilter

FILTER_REGISTRY: dict[str, type[VideoFilter]] = {
    "duration_resolution": DurationResolutionFilter,
    "text": TextCoverageFilter,
    "aesthetic": AestheticFilter,
    # "watermark": WatermarkFilter,
    "black_border": BlackBorderFilter,
    "overexposure": OverexposureFilter,
    "blur": BlurFilter,
    "motion_stability": MotionStabilityFilter,
    "nsfw": NSFWFilter,
}

__all__ = [
    "FilterResult",
    "VideoFilter",
    "DurationResolutionFilter",
    "TextCoverageFilter",
    "AestheticFilter",
    # "WatermarkFilter",
    "BlackBorderFilter",
    "OverexposureFilter",
    "BlurFilter",
    "MotionStabilityFilter",
    "NSFWFilter",
    "FILTER_REGISTRY",
]
