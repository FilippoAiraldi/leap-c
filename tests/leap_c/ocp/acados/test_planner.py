from typing import get_args

from leap_c.ocp.acados.planner import (
    TO_ACADOS_DIFFMPC_SENSOPTS,
    AcadosDiffMpcSensitivityOptions,
    SensitivityOptions,
)


def test_TO_ACADOS_DIFFMPC_SENSOPTS_is_bijective():
    """Tests that we did not forget about any sensitivity options, and that we did not
    repeat any mappings."""

    mapping = TO_ACADOS_DIFFMPC_SENSOPTS.copy()
    reverse = {v: k for k, v in mapping.items()}

    assert all(opt in mapping for opt in get_args(SensitivityOptions)), (
        "Some `SensitivityOptions` are missing from `TO_ACADOS_DIFFMPC_SENSOPTS`"
    )
    assert all(opt in reverse for opt in get_args(AcadosDiffMpcSensitivityOptions)), (
        "Some `AcadosDiffMpcSensitivityOptions` are missing from `TO_ACADOS_DIFFMPC_SENSOPTS`"
    )

    for dictionary in (mapping, reverse):
        values = list(dictionary.values())
        assert len(values) == len(set(values)), "Not bijective mapping detected!"
