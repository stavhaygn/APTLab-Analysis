import logging
from typing import Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple
from pathlib import Path
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType, EdgeType
from aptlab_analysis.dataset.utils import (
    load_node_csv,
    load_edge_csv,
    load_provenance_graph_schema,
)
from aptlab_analysis.dataset.exceptions import CSVLoadError
from aptlab_analysis.encoders import (
    Encoder,
    SequenceEncoder,
    AttackTechniqueClassEncoder,
)

FORMAT = "%(name)s %(levelname)s %(asctime)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("ProvenanceGraph")
logger.setLevel(logging.INFO)


class ProvenanceGraph(object):
    def __init__(
        self,
        csv_root_path: str = "csv/stavh.win11.normal20220105",
        schema_path: str = "schema/provenance_graph.yml",
        encoder_instance_mapping: Mapping[str, Encoder] = {},
    ):
        self._data = HeteroData()
        self._metadata = {}
        self._csv_root_path = Path(csv_root_path)
        self._schema = load_provenance_graph_schema(schema_path)
        self._encoder_instance_mapping = encoder_instance_mapping

        self._silent = True

    @property
    def data(self):
        return self._data

    @property
    def schema(self):
        return self._schema

    @property
    def encoders(self) -> Mapping[str, Encoder]:
        return self._encoder_instance_mapping

    @property
    def metadata(self):
        return self._metadata

    def _print_log(self, message: str, level: int = logging.INFO):
        if not self._silent:
            logger.log(level, message)

    def choose_nodes(
        self, indexes: Iterable[int], data: Optional[HeteroData] = None
    ) -> HeteroData:
        data = HeteroData() if data is None else data.clone()
        for index in indexes:
            node_type = self.metadata["nodes"].get(index, None)
            if node_type in self.data.node_types:
                data[node_type].x = self.data[node_type].x.clone()
        return data

    def choose_edges(
        self, indexes: Iterable[int], data: Optional[HeteroData] = None
    ) -> HeteroData:
        data = HeteroData() if data is None else data.clone()
        for index in indexes:
            edge_type = self.metadata["edges"].get(index, None)
            if edge_type in self.data.edge_types:
                data[edge_type].edge_index = self.data[edge_type].edge_index.clone()
                data[edge_type].edge_label = self.data[edge_type].edge_label.clone()
        return data

    def choose_nodes_and_edges(
        self, node_indexes: Iterable[int], edge_indexes: Iterable[int]
    ) -> HeteroData:
        data = HeteroData()
        data = self.choose_nodes(node_indexes, data)
        data = self.choose_edges(edge_indexes, data)
        return data

    def _parse_encoder_name_set_in_schema_with(self, unit: str) -> Set[str]:
        encoder_name_set = set()
        for unit_attr in self.schema[unit].values():
            column_encoder_mapping = unit_attr.get("encoders", {})
            encoder_names = list(column_encoder_mapping.values())
            encoder_name_set.update(encoder_names)
        return encoder_name_set

    def _parse_encoder_name_set_in_schema(self) -> Set[str]:
        encoder_name_set = self._parse_encoder_name_set_in_schema_with("nodes")
        encoder_name_set.update(self._parse_encoder_name_set_in_schema_with("edges"))
        return encoder_name_set

    def _create_encoder_instance_mapping(self) -> Dict[str, Encoder]:
        encoder_name_set = self._parse_encoder_name_set_in_schema()

        encoder_create_function_mapping = {
            "AttackTechniqueClassEncoder": AttackTechniqueClassEncoder,
            "SequenceEncoder": SequenceEncoder,
        }
        encoder_instance_mapping = {}

        for encoder_name in encoder_name_set:
            try:
                encoder_instance_mapping[
                    encoder_name
                ] = encoder_create_function_mapping[encoder_name]()
            except KeyError:
                raise KeyError(
                    f"There is no encoder named '{encoder_name}' in encoder_create_function_mapping"
                )

        return encoder_instance_mapping

    def _inject_encoder_instance(
        self, column_encoders_mapping: Mapping[str, str]
    ) -> Dict[str, Encoder]:
        column_encoder_instances_mapping = {}
        for column, encoder_name in column_encoders_mapping.items():
            try:
                column_encoder_instances_mapping[
                    column
                ] = self._encoder_instance_mapping[encoder_name]
            except KeyError:
                raise KeyError(
                    f"There is no encoder named '{encoder_name}' in encoder_instance_mapping"
                )
        return column_encoder_instances_mapping

    def _generate_csv_path(self, *units) -> Path:
        filename = "_".join(units)
        csv_filename = f"{filename}.csv"
        csv_path = self._csv_root_path / csv_filename
        return csv_path

    def _generate_column_encoder_instance_mapping(
        self, unit_attr: Mapping[str, Mapping[str, str]]
    ):
        column_encoder_instance_mapping = {}
        if "encoders" in unit_attr:
            column_encoder_instance_mapping = self._inject_encoder_instance(
                unit_attr["encoders"]
            )
        return column_encoder_instance_mapping

    def _load_nodes(
        self,
    ) -> Iterator[Tuple[NodeType, Optional[Tensor], Dict[str, int]]]:
        node_type_to_attr = self.schema["nodes"]

        for node_type, node_attr in node_type_to_attr.items():
            node_csv_path = self._generate_csv_path(node_type)

            column_encoder_instance_mapping = (
                self._generate_column_encoder_instance_mapping(node_attr)
            )

            try:
                node_x, node_identity_mapping = load_node_csv(
                    node_csv_path,
                    node_attr["identity_column"],
                    column_encoder_instance_mapping,
                )
                self._print_log(f"CSV file '{node_csv_path}' is loaded")
                yield node_type, node_x, node_identity_mapping

            except CSVLoadError as err:
                self._print_log(str(err), logging.WARNING)

    def _load_edges(
        self, node_type_to_identity_mapping: Mapping[str, Mapping[str, int]]
    ) -> Iterator[Tuple[EdgeType, Tensor, Optional[Tensor]]]:
        node_type_to_attr = self.schema["nodes"]
        edge_type_to_attr = self.schema["edges"]

        for edge_attr in edge_type_to_attr.values():
            src_node_type = edge_attr["source_node_type"]
            edge_type = edge_attr["edge_type"]
            dst_node_type = edge_attr["destination_node_type"]

            underscore_edge_type = edge_type.replace(" ", "_")
            edge_csv_path = self._generate_csv_path(
                src_node_type, underscore_edge_type, dst_node_type
            )

            column_encoder_instance_mapping = (
                self._generate_column_encoder_instance_mapping(edge_attr)
            )

            if not {src_node_type, dst_node_type}.issubset(
                node_type_to_identity_mapping
            ):
                self._print_log(
                    f"Node type '{src_node_type}' or '{dst_node_type}' not in identity_mapping"
                )
                continue

            src_node_identity_column = edge_attr.get(
                "src_identity_column",
                node_type_to_attr[src_node_type]["identity_column"],
            )

            dst_node_identity_column = edge_attr.get(
                "dst_identity_column",
                node_type_to_attr[dst_node_type]["identity_column"],
            )

            try:
                edge_index, edge_label = load_edge_csv(
                    edge_csv_path,
                    src_node_identity_column,
                    node_type_to_identity_mapping[src_node_type],
                    dst_node_identity_column,
                    node_type_to_identity_mapping[dst_node_type],
                    column_encoder_instance_mapping,
                )
                self._print_log(f"CSV file '{edge_csv_path}' is loaded")
                _edge_type = (src_node_type, edge_type, dst_node_type)
                yield _edge_type, edge_index, edge_label

            except CSVLoadError as err:
                self._print_log(str(err), logging.WARNING)

    @encoders.setter
    def encoders(self, encoders: Mapping[str, Encoder]) -> None:
        for encoder in encoders.values():
            if isinstance(encoder, Encoder):
                raise Exception(
                    f"'{type(encoder).__name__}'class should inherit from parent class 'aptlab_analysis.encoders.Encoder'"
                )
        self._encoder_instance_mapping = encoders

    def load(self, silent: bool = True) -> HeteroData:
        self._silent = silent
        if not self._encoder_instance_mapping:
            self._encoder_instance_mapping = self._create_encoder_instance_mapping()

        data = HeteroData()
        node_metadata = {}
        edge_metadata = {}

        node_type_to_identity_mapping = {}
        for index, (node_type, node_x, node_identity_mapping) in enumerate(
            self._load_nodes()
        ):
            node_metadata[index] = node_type
            if node_x is not None:
                data[node_type].x = node_x  # type: ignore
            node_type_to_identity_mapping[node_type] = node_identity_mapping

        for index, (edge_type, edge_index, edge_label) in enumerate(
            self._load_edges(node_type_to_identity_mapping)
        ):
            edge_metadata[index] = edge_type
            data[edge_type].edge_index = edge_index  # type: ignore
            if edge_label is not None:
                data[edge_type].edge_label = edge_label  # type: ignore

        self._metadata = {"nodes": node_metadata, "edges": edge_metadata}
        self._data = data.clone()

        return data
