from aptlab_analysis.data.utils import (
    deconstruct_technique_id_and_name,
    parse_sysmon_config,
    parse_technique_statements,
    generate_mapping_with_identity_as_key,
    generate_mapping_with_index_as_key,
)


def test_load_sysmon_config():
    pass


def test_deconstruct_technique_id_and_name():
    technique_id, technique_name = deconstruct_technique_id_and_name(
        "technique_id=T1055.012,technique_name=Process Hollowing"
    )
    assert (technique_id, technique_name) == ("T1055.012", "Process Hollowing")


def test_parse_technique_statements():
    technique_statements = parse_technique_statements(
        """
        <OriginalFileName name="technique_id=T1218.002,technique_name=rundll32.exe" condition="contains">rundll32.exe</OriginalFileName>
        <OriginalFileName name="technique_id=T1546.008,technique_name=Windows Error Reporting" condition="contains">werfault.exe</OriginalFileName>
        <OriginalFileName name="technique_id=T1016,technique_name=System Network Configuration Discovery" condition="is">ipconfig.exe</OriginalFileName>
        <OriginalFileName name="technique_id=T1016,technique_name=System Network Configuration Discovery" condition="is">route.exe</OriginalFileName>
    """
    )

    assert technique_statements == [
        "technique_id=T1016,technique_name=System Network Configuration Discovery",
        "technique_id=T1218.002,technique_name=rundll32.exe",
        "technique_id=T1546.008,technique_name=Windows Error Reporting",
    ]


def test_parse_sysmon_config():
    techniques = parse_sysmon_config(
        """
        <OriginalFileName name="technique_id=T1218.002,technique_name=rundll32.exe" condition="contains">rundll32.exe</OriginalFileName>
        <OriginalFileName name="technique_id=T1546.008,technique_name=Windows Error Reporting" condition="contains">werfault.exe</OriginalFileName>
        <OriginalFileName name="technique_id=T1016,technique_name=System Network Configuration Discovery" condition="is">ipconfig.exe</OriginalFileName>
        <OriginalFileName name="technique_id=T1016,technique_name=System Network Configuration Discovery" condition="is">route.exe</OriginalFileName>
    """
    )
    assert tuple(techniques) == (
        ("T1016", "System Network Configuration Discovery"),
        ("T1218.002", "rundll32.exe"),
        ("T1546.008", "Windows Error Reporting"),
    )


def test_generate_mapping_with_identity_as_key():
    identities = ["T1016", "T1218.002", "T1546.008"]
    mapping = generate_mapping_with_identity_as_key(identities)
    assert mapping == {
        "T1016": 0,
        "T1218.002": 1,
        "T1546.008": 2,
    }


def test_generate_mapping_with_identity_as_key_has_None_Class():
    identities = ["T1016", "T1218.002", "T1546.008"]
    mapping = generate_mapping_with_identity_as_key(identities, has_None_class=True)
    assert mapping == {
        "None": 0,
        "T1016": 1,
        "T1218.002": 2,
        "T1546.008": 3,
    }


def test_generate_mapping_with_index_as_key():
    identities = ["T1016", "T1218.002", "T1546.008"]
    mapping = generate_mapping_with_index_as_key(identities)
    assert mapping == {
        0: "T1016",
        1: "T1218.002",
        2: "T1546.008",
    }


def test_generate_mapping_with_index_as_key_has_None_class():
    identities = ["T1016", "T1218.002", "T1546.008"]
    mapping = generate_mapping_with_index_as_key(identities, has_None_class=True)
    assert mapping == {
        0: "None",
        1: "T1016",
        2: "T1218.002",
        3: "T1546.008",
    }
