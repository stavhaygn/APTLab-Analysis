import numpy as np
import torch
from aptlab_analysis.encoders import AttackTechniqueClassEncoder


def test_attack_technique_class_encoder():
    values = np.array(["T1055.012", "T1003", "T1055", "nan"], dtype=object)
    sysmon_config_path = "config/test_sysmonconfig.xml"
    attack_technique_class_encoder = AttackTechniqueClassEncoder(sysmon_config_path)

    x = attack_technique_class_encoder(values)
    assert x.equal(
        torch.tensor(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=torch.long,
        )
    )


def test_attack_technique_class_encoder_has_none_class():
    values = np.array(["T1055.012", "T1003", "T1055", "nan"], dtype=object)
    sysmon_config_path = "config/test_sysmonconfig.xml"
    attack_technique_class_encoder = AttackTechniqueClassEncoder(
        sysmon_config_path, has_None_class=True
    )

    x = attack_technique_class_encoder(values)
    assert x.equal(
        torch.tensor(
            [
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    )
