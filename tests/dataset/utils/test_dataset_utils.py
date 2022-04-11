import torch
from aptlab_analysis.dataset.utils import (
    load_provenance_graph_schema,
    load_node_csv,
    load_edge_csv,
)
from aptlab_analysis.encoders import ClassEncoder


def test_load_provenance_graph_schema():
    pass


def test_load_node_csv():
    integrity_level_mapping = {"Low": 0, "Medium": 1, "High": 2, "System": 3}
    column_encoder_mapping = {
        "integrity_level": ClassEncoder(integrity_level_mapping),
    }

    node_x, node_identity_mapping = load_node_csv(
        node_csv_path="csv/test/process.csv",
        identity_column="processId",
        column_encoder_mapping=column_encoder_mapping,
    )
    assert node_x is not None
    assert node_x.equal(torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]]))
    assert node_identity_mapping == {"43": 0, "57": 1}


def test_load_edge_csv():
    process_identity_mapping = {"43": 0, "57": 1}
    module_identity_mapping = {"412": 0}
    technique_id_mapping = {"T1016": 0, "T1033": 1, "T1047": 2}
    column_encoder_mapping = {"rule_technique_id": ClassEncoder(technique_id_mapping)}

    edge_index, edge_attr = load_edge_csv(
        edge_csv_path="csv/test/process_loaded_module.csv",
        src_node_identity_column="processId",
        src_node_identity_mapping=process_identity_mapping,
        dst_node_identity_column="moduleId",
        dst_node_identity_mapping=module_identity_mapping,
        column_encoder_mapping=column_encoder_mapping,
    )

    assert edge_index.equal(torch.tensor([[1, 0], [0, 0]]))
    assert edge_attr is not None
    assert edge_attr.equal(torch.tensor([[0, 0, 1], [0, 0, 1]]))
