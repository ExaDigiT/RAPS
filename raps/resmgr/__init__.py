"""
ResourceManager package initializer.
Exports a factory that returns the appropriate manager based on config.
"""
from .whole_node import WholeNodeResourceManager
from .multitenant import MultiTenantResourceManager


def make_resource_manager(total_nodes, down_nodes, config):
    """
    Factory to choose between whole-node and multitenant managers.
    """
    if config.get("multitenant", False):
        return MultiTenantResourceManager(total_nodes, down_nodes, config)
    return WholeNodeResourceManager(total_nodes, down_nodes, config)

# Alias for backward compatibility
ResourceManager = make_resource_manager

__all__ = [
    "make_resource_manager", 
    "ResourceManager", 
    "WholeNodeResourceManager", 
    "MultiTenantResourceManager"
]
