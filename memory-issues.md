# Episode Workflow Issues Log

Date: 2026-06-27

## Issues Noted

1. Episodes are too long for the current still-image workflow.
2. Scene pacing is too slow because the script target is set too high.
3. Subtitles are being burned into the final video as white text.
4. Thumbnail generation is drawing a white title overlay on top of the art.
5. Scene planning still exposes a `textOverlay` field, which could reintroduce text on visuals later.
6. The story review also called out stronger retention pacing, more motion, and shorter scenes.

## Fixes Applied

- Lowered the script target duration so the pipeline produces shorter episodes.
- Turned off burned subtitles by default so subtitles stay in files instead of being rendered onto the video.
- Added a thumbnail flag to suppress all text overlays.
- Forced scene `textOverlay` values to stay empty in the planner.

## Still To Review

- Whether we want to tighten the retention prompt further for even faster scene changes.
- Whether the thumbnail should stay completely text-free, or keep only an episode badge later.
