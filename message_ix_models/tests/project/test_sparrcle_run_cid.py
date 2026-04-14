import sys
from types import ModuleType, SimpleNamespace

from message_ix_models.project.sparrcle.scenario_generator import run_cid


def test_run_cid_forwards_reduction(monkeypatch) -> None:
    captured = {}

    def fake_run_cid_pipeline(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    fake_module = ModuleType("cid_pipeline")
    setattr(fake_module, "run_cid_pipeline", fake_run_cid_pipeline)
    monkeypatch.setitem(sys.modules, "cid_pipeline", fake_module)

    config = {
        "starters": [
            {
                "model": "SSP_SSP2_v6.5_sp",
                "scenario": "NPiREF_CI_0",
                "ssp": "SSP2",
                "magicc_output_dir": "_staging/magicc_output/example",
            }
        ],
        "cid": {
            "steps": ["buildings", "cooling"],
            "n_runs": 600,
            "reduction": "joint_p50",
        },
    }

    result = run_cid(
        mp=SimpleNamespace(),  # type: ignore[arg-type]
        model="SSP_SSP2_v6.5_sp",
        source_scenario="NPiREF_CI_0",
        config=config,  # type: ignore[arg-type]
        ssp="SSP2",
    )

    assert result == {"ok": True}
    assert captured["reduction"] == "joint_p50"
    assert captured["steps"] == ["buildings", "cooling"]
    assert captured["n_runs"] == 600
