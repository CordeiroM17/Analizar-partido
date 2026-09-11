"""
Pipeline de video de Analizar-partido.

Re-exporta lo que usan los notebooks para no tener que acordarse de en qué
submódulo vive cada función:

    from src.video import (
        check_environment, environment_ready, process_video,
        load_tracking, track_summary, players_per_frame,
        compare_with_lineups, positions_plausibility,
    )
"""

from src.video.pipeline import check_environment, environment_ready, process_video
from src.video.schema import load_tracking, save_tracking, validate_positions_df
from src.video.validation import (
    track_summary, players_per_frame, compare_with_lineups, positions_plausibility,
)

__all__ = [
    "check_environment", "environment_ready", "process_video",
    "load_tracking", "save_tracking", "validate_positions_df",
    "track_summary", "players_per_frame", "compare_with_lineups",
    "positions_plausibility",
]
