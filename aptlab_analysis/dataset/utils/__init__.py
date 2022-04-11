from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import yaml
import pandas as pd
from pandas.errors import EmptyDataError
import torch
from torch import Tensor
from aptlab_analysis.data.utils import generate_mapping_with_identity_as_key
from aptlab_analysis.encoders import Encoder
from aptlab_analysis.encoders.utils import encode_data_with_column_and_concat
from aptlab_analysis.dataset.exceptions import CSVLoadError


def load_provenance_graph_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path) as f:
        provenance_graph_schema = yaml.safe_load(f)
    return provenance_graph_schema


def load_node_csv(
    node_csv_path: Union[Path, str],
    identity_column: str,
    column_encoder_mapping: Optional[Mapping[str, Encoder]] = None,
) -> Tuple[Optional[Tensor], Dict[str, int]]:
    try:
        node_df = pd.read_csv(node_csv_path, index_col=identity_column)
    except (EmptyDataError, FileNotFoundError):
        raise CSVLoadError(node_csv_path)

    node_identity_mapping = generate_mapping_with_identity_as_key(
        tuple(node_df.index.unique().astype("str"))
    )

    node_x = None
    if column_encoder_mapping is not None:
        node_x = encode_data_with_column_and_concat(node_df, column_encoder_mapping)

    return node_x, node_identity_mapping


def load_edge_csv(
    edge_csv_path: Union[Path, str],
    src_node_identity_column: str,
    src_node_identity_mapping: Mapping[str, int],
    dst_node_identity_column: str,
    dst_node_identity_mapping: Mapping[str, int],
    column_encoder_mapping: Optional[Mapping[str, Encoder]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    try:
        edge_df = pd.read_csv(edge_csv_path)
    except (EmptyDataError, FileNotFoundError):
        raise CSVLoadError(edge_csv_path)

    src_node_indexes = [
        src_node_identity_mapping[node_identity]
        for node_identity in edge_df[src_node_identity_column].astype("str")
    ]
    dst_node_indexes = [
        dst_node_identity_mapping[node_identity]
        for node_identity in edge_df[dst_node_identity_column].astype("str")
    ]

    edge_index = torch.tensor([src_node_indexes, dst_node_indexes])

    edge_attr = None
    if column_encoder_mapping is not None:
        edge_attr = encode_data_with_column_and_concat(edge_df, column_encoder_mapping)

    return edge_index, edge_attr
