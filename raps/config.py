import json
import os
from typing import Dict, Any
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("RAPS_CONFIG", 'config')).resolve()


class ConfigManager:
    def __init__(self, system_name: str):
        self.config: Dict[str, Any] = {}
        self.load_system_config(system_name)
        self.system_name = system_name
        self.derive_values()

    def load_system_config(self, system_name: str) -> None:
        base_path = CONFIG_PATH / system_name
        config_files = ['system.json', 'power.json', 'scheduler.json']
        optional_files = ['cooling.json', 'uq.json', 'network.json']
        
        for config_file in config_files + optional_files:
            file_path = base_path / config_file
            if config_file in optional_files and not file_path.exists():
                continue  # Skip loading if the file is optional and doesn't exist
            if not file_path.exists():
                raise FileNotFoundError(f"Mandatory configuration file {config_file} not found.")
            config_data = self.load_config_file(file_path)
            self.config.update(config_data)
        
    @staticmethod
    def load_config_file(file_path: Path) -> dict[str, Any]:
        with open(file_path, 'r') as file:
            return json.load(file)

    def derive_values(self) -> None:
        # Derive SC_SHAPE and TOTAL_NODES
        num_cdus = self.config.get('NUM_CDUS', 0)
        racks_per_cdu = self.config.get('RACKS_PER_CDU', 0)
        nodes_per_rack = self.config.get('NODES_PER_RACK', 0)
        chassis_per_rack = self.config.get('CHASSIS_PER_RACK', 0)
        nodes_per_blade = self.config.get('NODES_PER_BLADE', 0)
        down_nodes = self.config.get('DOWN_NODES', 0)
        missing_racks = self.config.get('MISSING_RACKS', 0)

        self.config['NUM_RACKS'] = num_cdus * racks_per_cdu - len(missing_racks)
        self.config['SC_SHAPE'] = [num_cdus, racks_per_cdu, nodes_per_rack]
        self.config['TOTAL_NODES'] = num_cdus * racks_per_cdu * nodes_per_rack
        self.config['BLADES_PER_CHASSIS'] = int(nodes_per_rack / chassis_per_rack / nodes_per_blade)
        self.config['system_name'] = self.system_name

        # Generate POWER_DF_HEADER
        power_df_header = ["CDU"]
        for i in range(1, racks_per_cdu + 1):
            power_df_header.append(f"Rack {i}")
        power_df_header.append("Sum")
        for i in range(1, racks_per_cdu + 1):
            power_df_header.append(f"Loss {i}")
        power_df_header.append("Loss")
        self.config['POWER_DF_HEADER'] = power_df_header

        # Convert MISSING_RACKS into list of DOWN_NODES
        for rack in missing_racks:
            start_node_id = rack * nodes_per_rack
            end_node_id = start_node_id + nodes_per_rack
            down_nodes.extend(range(start_node_id, end_node_id))
        self.config['DOWN_NODES'] = down_nodes

        # Default multitenancy to False, unless explicitly set to True
        self.config['multitenant'] = bool(self.config.get("multitenant", False))

        self.config['AVAILABLE_NODES'] = self.config['TOTAL_NODES'] - len(down_nodes)

    def get(self, key: str) -> Any:
        return self.config.get(key)

    def get_config(self) -> Dict[str, Any]:
        # Return the complete config dictionary
        return self.config
