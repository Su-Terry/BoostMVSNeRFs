import imp

def make_network(cfg, preprocess=False):
    module = cfg.network_module
    path = cfg.network_path
    if preprocess:
        network = imp.load_source(module, path).Network(preprocess)
    else:
        network = imp.load_source(module, path).Network()
    return network
