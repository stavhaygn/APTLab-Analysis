from typing import Dict, Iterable, Iterator, List, Union, Tuple
import re


def load_sysmon_config(sysmon_config_path: str) -> str:
    with open(sysmon_config_path) as f:
        sysmon_config_content = f.read()
    return sysmon_config_content


def deconstruct_technique_id_and_name(technique_statement: str) -> Tuple[str, str]:
    technique_id_statement, technique_name_statement = technique_statement.split(",")
    _, technique_id = technique_id_statement.split("=")
    _, technique_name = technique_name_statement.split("=")
    return (technique_id, technique_name)


def parse_technique_statements(sysmon_config_content: str) -> List[str]:
    regex = re.compile(r"technique_id=T[^,]+,technique_name=[^\"]+")
    matches = regex.findall(sysmon_config_content)
    technique_statements = list(set(matches))
    technique_statements.sort()
    return technique_statements


def parse_sysmon_config(sysmon_config_content: str) -> Iterator[Tuple[str, str]]:
    technique_statements = parse_technique_statements(sysmon_config_content)
    techniques = (
        deconstruct_technique_id_and_name(statement)
        for statement in technique_statements
    )
    return techniques


def generate_mapping_with_identity_as_key(
    identities: Iterable[Union[int, str]], has_None_class: bool = False
) -> Dict[Union[int, str], int]:
    offset = 1 if has_None_class else 0
    mapping = {identity: index + offset for index, identity in enumerate(identities)}
    if has_None_class:
        mapping["None"] = 0
    return mapping


def generate_mapping_with_index_as_key(
    identities: Iterable[Union[int, str]], has_None_class: bool = False
) -> Dict[int, Union[int, str]]:
    offset = 1 if has_None_class else 0
    mapping = {index + offset: identity for index, identity in enumerate(identities)}
    if has_None_class:
        mapping[0] = "None"
    return mapping
