from worker.artifacts import main_clip_key


def test_main_clip_key_matches_pipeline_layout() -> None:
    assert (
        main_clip_key(
            "2d875543-871d-4f4d-bc14-181aea950492",
            "clip_04",
        )
        == "pipeline/2d875543-871d-4f4d-bc14-181aea950492/clips/clip_04.mp4"
    )
