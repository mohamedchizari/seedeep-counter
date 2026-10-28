"""Validation for video timing metadata used by count_video.py."""

LOWEST, HIGHEST = 0.1, 1000.0  # frames a second: one frame in ten seconds to a high-speed camera's rate


def effective_fps(reported):
    """Use 25 only when the decoder reports no frame rate (0). Any other rate outside LOWEST to HIGHEST frames a second is
    refused: a negative one, one that is not a number or infinite (no comparison with NaN holds), or one so small or large
    that every time and the video's length would be meaningless."""
    if reported == 0:
        return 25.0
    if not LOWEST <= reported <= HIGHEST:
        raise ValueError(f'video frame rate {reported!r} is outside {LOWEST:g} to {HIGHEST:,.0f} frames a second')
    return reported
