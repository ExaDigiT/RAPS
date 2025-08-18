"""
ResourceManager package initializer.
Exports a factory that returns the appropriate manager based on config.
"""
from .default import ExclusiveNodeResourceManager
from .multitenant import MultiTenantResourceManager


def make_resource_manager(total_nodes, down_nodes, config):
    """
    Factory to choose between exclusive-node and multitenant managers.
    """
    if config.get("multitenant", False):
        return MultiTenantResourceManager(total_nodes, down_nodes, config)
    return ExclusiveNodeResourceManager(total_nodes, down_nodes, config)


# Alias for backward compatibility
ResourceManager = make_resource_manager

__all__ = [
    "make_resource_manager",
    "ResourceManager",
    "ExclusiveNodeResourceManager",
    "MultiTenantResourceManager"
]
