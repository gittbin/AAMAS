"""Physical contract and frozen-evaluator isolation for the disclosed overflow fix."""
import torch
import pytest
from experiments_numerical_erratum.power import corrected_decoder, corrected_env_for, legacy_decode
from experiments_paper_completion.engine import config_for, env_for
from momgrap.marl.actions.decoder import _enforce_floor


@pytest.mark.parametrize("gains", [[1e-12,2e-12,3e-12,4e-12], [0,1e-20,1e-12,1e-10], [0,0,0,0]])
def test_power_floor_sum_and_reference(gains):
    evo = config_for("feedforward",305,400384); cfg=evo.physical
    H=torch.zeros(1,1,4,16,dtype=cfg.dtype_complex)
    W=torch.zeros(1,1,16,4,dtype=cfg.dtype_complex)
    for k,g in enumerate(gains): H[0,0,k,k]=g**0.5; W[0,0,k,k]=1
    events=[]; decode=corrected_decoder(events)
    for index in range(9):
        old=legacy_decode(index,H,W,cfg,evo.actions)
        actual=decode(index,H,W,cfg,evo.actions)
        assert torch.isfinite(actual).all()
        assert actual.min() >= 0.05*cfg.p_max_watt/4-1e-8
        assert torch.isclose(actual.sum(),torch.tensor(cfg.p_max_watt,dtype=cfg.dtype_real),rtol=1e-6)
        if torch.isfinite(old).all(): assert torch.equal(actual,old)
        else:
            g=(torch.einsum('nskm,nsmk->nsk',H,W).abs().square().mean((0,1)).clamp_min(1e-30)).double()
            w=(g.pow(-2)/g.pow(-2).sum()).to(cfg.dtype_real)
            reference=_enforce_floor(w*cfg.p_max_watt,cfg.p_max_watt,0.05*cfg.p_max_watt/4)
            assert torch.equal(actual.flatten(),reference)


def test_instance_globals_are_isolated():
    evo=config_for("feedforward",305,400384)
    original=env_for(evo,"feedforward",eval_mode=True)
    backend=original._base_step.__func__.__globals__["evaluate_control_interval"]
    original_decode=backend.__globals__["decode_power_action"]
    changed=corrected_env_for(evo,"feedforward",events=[],eval_mode=True)
    assert original._base_step.__func__.__globals__["evaluate_control_interval"] is backend
    assert backend.__globals__["decode_power_action"] is original_decode
    assert changed._base_step.__func__.__globals__["evaluate_control_interval"] is not backend
    original.close(); changed.close()
