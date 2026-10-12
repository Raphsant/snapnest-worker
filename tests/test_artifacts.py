from worker.artifacts import (
    clip_caption_key,
    edited_clip_caption_key,
    main_clip_key,
)


def test_main_clip_key_matches_pipeline_layout() -> None:
    assert (
        main_clip_key(
            "2d875543-871d-4f4d-bc14-181aea950492",
            "clip_04",
        )
        == "pipeline/2d875543-871d-4f4d-bc14-181aea950492/clips/clip_04.mp4"
    )


def test_clip_caption_key_matches_pipeline_layout() -> None:
    assert (
        clip_caption_key(
            "2d875543-871d-4f4d-bc14-181aea950492",
            "clip_04",
        )
        == "pipeline/2d875543-871d-4f4d-bc14-181aea950492/clips/captions/clip_04.srt"
    )


def test_edited_clip_caption_key_matches_pipeline_layout() -> None:
    assert (
        edited_clip_caption_key(
            "2d875543-871d-4f4d-bc14-181aea950492",
            "clip_04",
        )
        == "pipeline/2d875543-871d-4f4d-bc14-181aea950492/clips/captions/"
        "clip_04.edited.srt"
    )
