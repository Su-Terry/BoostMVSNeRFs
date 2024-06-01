from lib.config import cfg, args
import numpy as np
import os

def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))
    
def run_preprocess():
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils import net_utils
    import json

    print('Preprocessing mask...')
    
    network = make_network(cfg, preprocess=True).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()
    
    data_loader_train = make_data_loader(cfg, is_train=True)
    data_loader_test = make_data_loader(cfg, is_train=False)
    
    print('Preprocessing train set...')
    outputs_train = get_view_selection(data_loader_train, network)
    # outputs_train = {}
    print('Preprocessing test set...')
    outputs_test = get_view_selection(data_loader_test, network)
    outputs = {**outputs_train, **outputs_test}
    
    # dump outputs to json file
    view_selection_file = os.path.join(cfg.result_dir, f'view_selection.json')
    os.makedirs(cfg.result_dir, exist_ok=True)
    with open(view_selection_file, 'w') as f:
        json.dump(outputs, f)
        
def get_view_selection(data_loader, network):
    import tqdm
    import torch

    outputs = {}
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            output = network.forward_view_selection(batch)
            torch.cuda.synchronize()
        outputs.update(output)
    return outputs

def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output = network(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    evaluator.summarize()
    if len(net_time) > 1:
        print('FPS: ', 1./np.mean(net_time[1:]))
    else:
        print('FPS: ', 1./np.mean(net_time))


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.utils.data_utils import to_cuda

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch)
        visualizer.visualize(output, batch)
    visualizer.summarize()

if __name__ == '__main__':
    globals()['run_' + args.type]()
