from aptlab_analysis.data.utils import (
    load_sysmon_config,
    parse_sysmon_config,
    generate_mapping_with_identity_as_key,
    generate_mapping_with_index_as_key,
)
from aptlab_analysis.encoders import ClassEncoder


class AttackTechniqueClassEncoder(ClassEncoder):
    def __init__(
        self,
        sysmon_config_path: str = "config/sysmonconfig.xml",
        has_None_class: bool = True,
    ):
        sysmon_config = load_sysmon_config(sysmon_config_path)
        techniques = parse_sysmon_config(sysmon_config)
        technique_ids = list(set([technique_id for technique_id, _ in techniques]))
        technique_ids.sort()
        technique_mapping = generate_mapping_with_identity_as_key(
            technique_ids, has_None_class
        )
        index_mapping = generate_mapping_with_index_as_key(
            technique_ids, has_None_class
        )
        self.mapping = technique_mapping
        self.reverse_mapping = index_mapping
        self.has_None_class = has_None_class
