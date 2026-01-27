import torch
from utils import Cfg
from dataset import LBSNDataset
from layer import NeighborSampler
from model import STHGCN

cfg = Cfg('best_conf/csv_events.yml')
cfg.model_args.sizes = [int(x) for x in str(cfg.model_args.sizes).split('-')]
print('Loading dataset...')
dataset = LBSNDataset(cfg)
print('Dataset loaded.')

graph_tensors = []
if getattr(dataset, 'entity_graph', None) is not None:
    graph_tensors.append(('entity_graph.x', dataset.entity_graph.x))
if getattr(dataset, 'event_graph', None) is not None:
    graph_tensors.append(('event_graph.x', dataset.event_graph.x))
if getattr(dataset, 'chain_graph', None) is not None:
    graph_tensors.append(('chain_graph.x', dataset.chain_graph.x))

for name, tensor in graph_tensors:
    if tensor is None:
        print(name, 'is None')
        continue
    arr = tensor.float()
    col_min = arr.min(dim=0).values
    col_max = arr.max(dim=0).values
    col_mean = arr.mean(dim=0)
    print(f'[{name}] shape={tuple(arr.shape)}')
    for i, (mn, mx, mean) in enumerate(zip(col_min.tolist(), col_max.tolist(), col_mean.tolist())):
        print(f'  dim {i}: min={mn:.4f} max={mx:.4f} mean={mean:.4f}')

edge_delta_names = ['entity2event', 'event2event', 'event2chain', 'chain2chain']
for name, tensor in zip(edge_delta_names, dataset.edge_delta_t):
    if tensor is None:
        print(f'[edge_delta_t] {name}: None')
        continue
    vals = tensor.float().view(-1)
    if vals.numel() == 0:
        print(f'[edge_delta_t] {name}: empty tensor')
        continue
    print(f'[edge_delta_t] {name}: shape={tuple(tensor.shape)} min={vals.min().item():.6f} max={vals.max().item():.6f} mean={vals.mean().item():.6f}')

train_loader = NeighborSampler(
    x=dataset.x_for_sampler,
    edge_index=dataset.edge_index,
    edge_attr=dataset.edge_attr,
    edge_t=dataset.edge_t,
    edge_delta_t=dataset.edge_delta_t,
    edge_delta_s=dataset.edge_delta_s,
    edge_type=dataset.edge_type,
    sizes=cfg.model_args.sizes,
    sample_idx=dataset.sample_idx_train,
    node_idx=dataset.node_idx_train,
    label=dataset.labels_train,
    candidates=dataset.candidates_train,
    max_time=dataset.max_time_train,
    total_nodes=getattr(dataset, 'total_nodes', dataset.x_for_sampler.size(0)),
    batch_size=cfg.run_args.batch_size,
    shuffle=False,
    num_workers=0,
    intra_jaccard_threshold=getattr(cfg.model_args, 'intra_jaccard_threshold', 0.0),
    inter_jaccard_threshold=getattr(cfg.model_args, 'inter_jaccard_threshold', 0.0),
)

batch = next(iter(train_loader))
print('First batch sample_idx shape:', batch.sample_idx.shape)
print('First row candidates:', batch.candidates[0].tolist())
print('First row label:', int(batch.y[0].item()))
print('Label equals candidates[0,0]? ', int(batch.candidates[0,0].item()) == int(batch.y[0].item()))

model = STHGCN(cfg, dataset)
model.eval()
with torch.no_grad():
    out = model(batch)
print('gold_scores shape:', out['gold_scores'].shape)
print('negative_scores shape:', out['negative_scores'].shape)
print('First 5 gold scores:', out['gold_scores'].squeeze()[:5])
if out['negative_scores'].numel() > 0:
    print('First row negative scores (first 5):', out['negative_scores'][0][:5])
