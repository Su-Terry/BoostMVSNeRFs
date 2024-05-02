import os
import imp

def make_mcp(cfg):
    module = cfg.mcp_module
    path = cfg.mcp_path
    mcp = imp.load_source(module, path).MCP()
    return mcp

def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network
