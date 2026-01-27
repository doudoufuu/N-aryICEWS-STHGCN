import argparse   
import logging   
import os   
import os.path as osp   
import glob  
import random   
import sys   
from datetime import datetime   

import torch   
from torch.utils.tensorboard import SummaryWriter   
from tqdm import tqdm   

from dataset import LBSNDataset   
from layer import NeighborSampler   
from model import STHGCN   
from preprocess import preprocess   

from utils import (   
    Cfg,   
    count_parameters,   
    save_model,   
    seed_torch,   
    set_logger,   
    test_step,   
    get_root_dir,
)   

# Optional plotting support (matplotlib)
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


def build_sampler(cfg, dataset, split: str, is_train: bool) -> NeighborSampler:   
    sample_idx = getattr(dataset, f"sample_idx_{split}")   
    node_idx = getattr(dataset, f"node_idx_{split}")   
    labels = getattr(dataset, f"labels_{split}")   
    candidates = getattr(dataset, f"candidates_{split}")   
    max_time = getattr(dataset, f"max_time_{split}")   

    return NeighborSampler(   
        x=dataset.x_for_sampler,   
        edge_index=dataset.edge_index,   
        edge_attr=dataset.edge_attr,   
        edge_t=dataset.edge_t,   
        edge_delta_t=dataset.edge_delta_t,   
        edge_delta_s=dataset.edge_delta_s,   
        edge_type=dataset.edge_type,   
        sizes=cfg.model_args.sizes,   
        sample_idx=sample_idx,   
        node_idx=node_idx,   
        label=labels,   
        candidates=candidates,   
        max_time=max_time,   
        total_nodes=getattr(dataset, "total_nodes", dataset.x_for_sampler.size(0)),   
        batch_size=cfg.run_args.batch_size if is_train else cfg.run_args.eval_batch_size,   
        shuffle=is_train,   
        num_workers=cfg.run_args.num_workers if is_train else 0,   
        pin_memory="cuda" in cfg.run_args.device if hasattr(cfg.run_args, "device") else False,   
        drop_last=False,   
        intra_jaccard_threshold=getattr(cfg.model_args, "intra_jaccard_threshold", 0.0),   
        inter_jaccard_threshold=getattr(cfg.model_args, "inter_jaccard_threshold", 0.0),   
    )   


def prepare_hparams(cfg, seed: int) -> dict:   
    hparam_dict = {}   
    for group in cfg.__dict__.values():   
        hparam_dict.update(group.__dict__)   
    hparam_dict["seed"] = seed   
    hparam_dict["sizes"] = "-".join(str(item) for item in cfg.model_args.sizes)   
    return hparam_dict   


def maybe_add_stream_handler():   
    root_logger = logging.getLogger()   
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):   
        handler = logging.StreamHandler(sys.stdout)   
        handler.setLevel(logging.INFO)   
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))   
        root_logger.addHandler(handler)   


def main():   
    parser = argparse.ArgumentParser()   
    parser.add_argument("-f", "--yaml_file", required=True, help="The configuration file.")   
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Directory containing checkpoint.pt used to resume training."
    )
    args = parser.parse_args()   

    cfg = Cfg(args.yaml_file)   
    cfg.model_args.sizes = [int(size) for size in str(cfg.model_args.sizes).split("-")]   

    cli_init_ckpt = args.init_checkpoint.strip() if isinstance(args.init_checkpoint, str) else None
    if cli_init_ckpt:
        cfg.run_args.init_checkpoint = cli_init_ckpt
    elif not hasattr(cfg.run_args, "init_checkpoint"):
        cfg.run_args.init_checkpoint = None

    if int(cfg.run_args.gpu) >= 0 and torch.cuda.is_available():
        device = f"cuda:{cfg.run_args.gpu}"
        # 进行一次 CUDA 预热，若失败则回退到 CPU
        try:
            torch.cuda.init()
            # 触发一次极小线性层计算以初始化 cuBLAS
            lin = torch.nn.Linear(4, 4).to(device)
            x_test = torch.randn(2, 4, device=device)
            with torch.no_grad():
                _ = lin(x_test)
        except Exception as e:
            print(f"[Warn] CUDA 预热失败，回退到 CPU。原因: {e}")
            device = "cpu"
    else:
        device = "cpu"
    cfg.run_args.device = device

    if cfg.run_args.seed is None:   
        seed = random.randint(0, 100000000)   
    else:   
        seed = int(cfg.run_args.seed)   
    seed_torch(seed)   

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")   
    cfg.run_args.save_path = osp.join("tensorboard", current_time, cfg.dataset_args.dataset_name)   
    cfg.run_args.log_path = osp.join("log", current_time, cfg.dataset_args.dataset_name)   
    os.makedirs(cfg.run_args.save_path, exist_ok=True)   
    os.makedirs(cfg.run_args.log_path, exist_ok=True)   

    set_logger(cfg.run_args)   
    maybe_add_stream_handler()   
    summary_writer = SummaryWriter(log_dir=cfg.run_args.save_path)   
    hparam_dict = prepare_hparams(cfg, seed)   

    # Optional: clean stale caches outside preprocess when force_rebuild=True
    if bool(getattr(cfg.run_args, "force_rebuild", False)):
        try:
            root_dir = get_root_dir()
        except Exception:
            root_dir = os.getcwd()
        data_path_cfg = str(getattr(cfg.dataset_args, "data_path", "")).strip()
        if data_path_cfg:
            preprocessed_path = data_path_cfg if osp.isabs(data_path_cfg) else osp.join(root_dir, data_path_cfg)
        else:
            preprocessed_path = osp.join(root_dir, 'data', cfg.dataset_args.dataset_name, 'preprocessed_3')
        patterns = [
            'entity_graph.pt',
            'event_graph.pt',
            'chain_graph.pt',
            'hypergraph_checkpoint.pkl',
            'hypergraph_checkpoint_data.pkl',
            'event_edges_threshold*.pt',
            'event_edges_ckpt.pkl',
            'event_edge_deltas.pt',
            'l1_relation_features_checkpoint.pkl',
        ]
        removed = 0
        for pat in patterns:
            for fp in glob.glob(osp.join(preprocessed_path, pat)):
                try:
                    os.remove(fp)
                    removed += 1
                    print(f"[Preprocess] 清理缓存: {fp}")
                except Exception:
                    pass
        print(f"[Preprocess] force_rebuild=True，已在 {preprocessed_path} 清理 {removed} 个缓存文件")

    preprocess(cfg)   

    logging.info("[Training] Building dataset...")   
    dataset = LBSNDataset(cfg)   
    cfg.dataset_args.spatial_slots = getattr(dataset, "spatial_slots", None)   
    cfg.dataset_args.num_event = getattr(dataset, "num_event", None)   

    train_loader = build_sampler(cfg, dataset, "train", is_train=True) if cfg.run_args.do_train else None   
    valid_loader = build_sampler(cfg, dataset, "valid", is_train=False) if cfg.run_args.do_validate else None   
    test_loader = build_sampler(cfg, dataset, "test", is_train=False) if cfg.run_args.do_test else None   

    logging.info("[Training] Building model...")   
    model = STHGCN(cfg, dataset).to(device)   
    logging.info(f"[Training] Seed: {seed}")   
    logging.info(f"[Training] #Parameters: {count_parameters(model)}")   

    best_metrics = float("-inf")   
    global_step = 0   

    # Loss histories for plotting
    train_step_hist = []
    train_loss_hist = []
    valid_step_hist = []
    valid_loss_hist = []

    if cfg.run_args.do_train:   
        current_lr = float(cfg.run_args.learning_rate)   
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_lr,
            weight_decay=float(getattr(cfg.run_args, "weight_decay", 0.0))
        )   
        warm_up_steps = int(cfg.run_args.warm_up_steps) if cfg.run_args.warm_up_steps else cfg.run_args.max_steps // 2   
        cooldown_rate = float(cfg.run_args.cooldown_rate) if cfg.run_args.cooldown_rate else 1.0   
        use_full_softmax = bool(getattr(cfg.run_args, "use_full_softmax", False))   

        if use_full_softmax and isinstance(getattr(dataset, "labels_train", None), torch.Tensor):   
            lb = dataset.labels_train.min().item()   
            ub = dataset.labels_train.max().item()   
            logging.info(   
                "[Training] Full-softmax enabled: train label range=[%d, %d], num_event=%d",   
                int(lb),   
                int(ub),   
                getattr(dataset, "num_event", -1),   
            )   

        if hasattr(cfg.run_args, "lr_decay_factor"):   
            decay_factor = float(cfg.run_args.lr_decay_factor)   
        else:   
            decay_factor = 0.5 if use_full_softmax else 0.1   
        if not (0.0 < decay_factor < 1.0):   
            logging.warning("[Training] Invalid lr_decay_factor=%.4f, fallback to %.2f", decay_factor, 0.5 if use_full_softmax else 0.1)   
            decay_factor = 0.5 if use_full_softmax else 0.1   

        if hasattr(cfg.run_args, "min_learning_rate"):   
            min_learning_rate = float(cfg.run_args.min_learning_rate)   
        else:   
            min_learning_rate = 1e-5 if use_full_softmax else 1e-7   

        if min_learning_rate <= 0.0 or min_learning_rate >= current_lr:   
            min_learning_rate = min(current_lr * 0.1, 1e-5 if use_full_softmax else 1e-7)   
            logging.warning("[Training] Resetting min_learning_rate to %.2e to keep it < current_lr.", min_learning_rate)   

        if hasattr(cfg.run_args, "max_lr_decays"):   
            max_lr_decays = int(cfg.run_args.max_lr_decays)   
        else:   
            max_lr_decays = 3 if use_full_softmax else 100   
        max_lr_decays = max(0, max_lr_decays)   

        next_lr_decay_step = warm_up_steps   
        num_lr_decays = 0   

        # Optional validation-based scheduler
        use_plateau_scheduler = bool(getattr(cfg.run_args, "use_plateau_scheduler", True))
        if use_plateau_scheduler:
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=min_learning_rate,
                verbose=True,
            )
        else:
            plateau_scheduler = None

        if cfg.run_args.init_checkpoint:   
            logging.info(f"[Training] Loading checkpoint from {cfg.run_args.init_checkpoint}")   
            checkpoint = torch.load(osp.join(cfg.run_args.init_checkpoint, "checkpoint.pt"), map_location=device)   
            global_step = checkpoint["step"]   
            model.load_state_dict(checkpoint["model_state_dict"])   
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])   
            current_lr = checkpoint["current_learning_rate"]   
            warm_up_steps = checkpoint["warm_up_steps"]   
            cooldown_rate = checkpoint["cooldown_rate"]   
            cfg.model_args.sizes = checkpoint["sizes"]   
            for param_group in optimizer.param_groups:   
                param_group["lr"] = current_lr   
            if "next_lr_decay_step" in checkpoint:   
                next_lr_decay_step = checkpoint["next_lr_decay_step"]   
            else:   
                next_lr_decay_step = warm_up_steps   
            if "num_lr_decays" in checkpoint:   
                num_lr_decays = int(checkpoint["num_lr_decays"])   
            if not hasattr(cfg.run_args, "lr_decay_factor") and "lr_decay_factor" in checkpoint:   
                decay_factor = float(checkpoint["lr_decay_factor"])   
            if not hasattr(cfg.run_args, "min_learning_rate") and "min_learning_rate" in checkpoint:   
                min_learning_rate = float(checkpoint["min_learning_rate"])   
            if not hasattr(cfg.run_args, "max_lr_decays") and "max_lr_decays" in checkpoint:   
                max_lr_decays = int(checkpoint["max_lr_decays"])   
        else:   
            logging.info("[Training] Randomly initializing model parameters.")   

        logging.info(f"[Training] Initial learning rate: {current_lr}")   

        for epoch in range(cfg.run_args.epoch):   
            if global_step >= cfg.run_args.max_steps:   
                break   

            training_losses = []   
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")   
            for batch in iterator:   
                if global_step >= cfg.run_args.max_steps:   
                    break   

                model.train()   
                batch = batch.to(device)   
                # Margin warm-up: linearly ramp margin parameters in the first N steps
                try:
                    warm_steps = int(getattr(cfg.run_args, "margin_warmup_steps", 5000))
                    ramp = 1.0 if warm_steps <= 0 else max(0.0, min(1.0, global_step / warm_steps))
                    if hasattr(model, "margin_value"):
                        base_margin = float(getattr(cfg.run_args, "margin", getattr(model, "margin_value", 0.2)))
                        model.margin_value = base_margin * ramp
                    if hasattr(model, "margin_weight"):
                        base_weight = float(getattr(cfg.run_args, "margin_weight", getattr(model, "margin_weight", 0.5)))
                        model.margin_weight = base_weight * ramp
                except Exception:
                    pass
                output = model(batch)   
                loss = output["loss"]   

                l2_reg_coeff = float(getattr(cfg.run_args, "l2_reg", 0.0))
                if l2_reg_coeff > 0.0:
                    l2_term = torch.zeros(1, device=device)
                    for param in model.parameters():
                        if param.requires_grad:
                            l2_term = l2_term + param.pow(2).sum()
                    loss = loss + l2_reg_coeff * l2_term
                training_losses.append(loss.item())   

                optimizer.zero_grad()   
                loss.backward()   
                # Gradient clipping to stabilize early training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()   

                summary_writer.add_scalar("train/loss_step", loss.item(), global_step)
                # Also log CE-only loss to align with validation CE
                try:
                    ce_t = output.get("ce_loss", None)
                    if ce_t is not None:
                        summary_writer.add_scalar("train/ce_loss_step", float(ce_t.item()), global_step)
                except Exception:
                    pass
                summary_writer.add_scalar("train/learning_rate", current_lr, global_step)
                try:
                    if isinstance(output.get("margin_mean"), torch.Tensor):
                        summary_writer.add_scalar("train/margin_mean", float(output["margin_mean"].item()), global_step)
                    if isinstance(output.get("hardest_gap"), torch.Tensor):
                        summary_writer.add_scalar("train/hardest_gap", float(output["hardest_gap"].item()), global_step)
                except Exception:
                    pass

                # collect train loss point
                train_step_hist.append(global_step)
                train_loss_hist.append(float(loss.item()))

                if cfg.run_args.do_validate and valid_loader is not None and global_step % cfg.run_args.valid_steps == 0:   
                    logging.info(f"[Evaluating] Epoch {epoch}, step {global_step}")   
                    # Disable margin loss during evaluation to make loss comparable across splits
                    _saved_margin_flag = bool(getattr(model, "margin_loss_enable", True))
                    try:
                        if hasattr(model, "margin_loss_enable"):
                            model.margin_loss_enable = False
                        eval_results = test_step(   
                            model, valid_loader, use_full_ranking=getattr(cfg.run_args, "eval_full_ranking", False)   
                        )   
                    finally:
                        if hasattr(model, "margin_loss_enable"):
                            model.margin_loss_enable = _saved_margin_flag
                    recalls, NDCGs, MAPs, mrr_res, eval_loss, hit_scores = eval_results   
                    hits_at_1 = hit_scores.get(1, 0.0) if isinstance(hit_scores, dict) else 0.0   
                    hits_at_5 = hit_scores.get(5, 0.0) if isinstance(hit_scores, dict) else 0.0   
                    hits_at_10 = hit_scores.get(10, 0.0) if isinstance(hit_scores, dict) else 0.0   
                    for k_val, value in recalls.items():   
                        summary_writer.add_scalar(f"validate/Recall@{k_val}", 100 * value, global_step)   
                    for k_val, value in NDCGs.items():   
                        summary_writer.add_scalar(f"validate/NDCG@{k_val}", value, global_step)   
                    for k_val, value in MAPs.items():   
                        summary_writer.add_scalar(f"validate/MAP@{k_val}", value, global_step)   
                    summary_writer.add_scalar("validate/MRR", mrr_res, global_step)   
                    summary_writer.add_scalar("validate/loss", eval_loss, global_step)   
                    summary_writer.add_scalar("validate/Hits@1", hits_at_1, global_step)   
                    summary_writer.add_scalar("validate/Hits@5", hits_at_5, global_step)   
                    summary_writer.add_scalar("validate/Hits@10", hits_at_10, global_step)   
                    logging.info(   
                        "[Evaluating] Hits@1/5/10 : %.6f / %.6f / %.6f",   
                        hits_at_1,   
                        hits_at_5,   
                        hits_at_10,   
                    )   

                    # Step ReduceLROnPlateau with validation loss
                    if use_plateau_scheduler and plateau_scheduler is not None:
                        try:
                            plateau_scheduler.step(float(eval_loss))
                            # keep current_lr in sync with optimizer
                            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else current_lr
                        except Exception:
                            pass

                    # collect valid loss point
                    valid_step_hist.append(global_step)
                    valid_loss_hist.append(float(eval_loss))

                    metrics_score = 4 * recalls.get(1, 0.0) + recalls.get(20, recalls.get(max(recalls.keys()), 0.0) if recalls else 0.0)   
                    if metrics_score > best_metrics:   
                        best_metrics = metrics_score   
                        save_payload = {   
                            "step": global_step,   
                            "current_learning_rate": current_lr,   
                            "warm_up_steps": warm_up_steps,   
                            "cooldown_rate": cooldown_rate,   
                            "sizes": cfg.model_args.sizes,   
                            "next_lr_decay_step": next_lr_decay_step,   
                            "num_lr_decays": num_lr_decays,   
                            "lr_decay_factor": decay_factor,   
                            "min_learning_rate": min_learning_rate,   
                            "max_lr_decays": max_lr_decays,   
                        }   
                        logging.info(f"[Training] Saving new best model at step {global_step}, epoch {epoch}.")   
                        save_model(model, optimizer, save_payload, cfg.run_args, hparam_dict)   

                # If ReduceLROnPlateau disabled, fall back to manual step-based decay (if allowed)
                if not use_plateau_scheduler and max_lr_decays > 0 and decay_factor < 1.0:
                    if (
                        global_step >= next_lr_decay_step
                        and num_lr_decays < max_lr_decays
                        and current_lr > min_learning_rate   
                    ):   
                        new_lr = max(current_lr * decay_factor, min_learning_rate)   
                        if new_lr < current_lr:   
                            current_lr = new_lr   
                            for param_group in optimizer.param_groups:   
                                param_group["lr"] = current_lr   
                            logging.info(f"[Training] Adjust learning rate to {current_lr} at step {global_step}.")   
                            num_lr_decays += 1   
                        next_lr_decay_step = int(next_lr_decay_step * cooldown_rate) if cooldown_rate > 1.0 else next_lr_decay_step + warm_up_steps   

                global_step += 1   

            if training_losses:   
                epoch_loss = sum(training_losses) / len(training_losses)   
                logging.info(f"[Training] Epoch {epoch} average loss: {epoch_loss}")   
                summary_writer.add_scalar("train/loss_epoch", epoch_loss, epoch)   

        if cfg.run_args.do_train and (not cfg.run_args.do_validate or best_metrics == float("-inf")):   
            save_payload = {   
                "step": global_step,   
                "current_learning_rate": current_lr if 'current_lr' in locals() else float(cfg.run_args.learning_rate),   
                "warm_up_steps": warm_up_steps if 'warm_up_steps' in locals() else cfg.run_args.max_steps // 2,   
                "cooldown_rate": cooldown_rate if 'cooldown_rate' in locals() else 1.0,   
                "sizes": cfg.model_args.sizes,   
                "next_lr_decay_step": next_lr_decay_step if 'next_lr_decay_step' in locals() else warm_up_steps,   
                "num_lr_decays": num_lr_decays if 'num_lr_decays' in locals() else 0,   
                "lr_decay_factor": decay_factor if 'decay_factor' in locals() else 0.1,   
                "min_learning_rate": min_learning_rate if 'min_learning_rate' in locals() else 1e-7,   
                "max_lr_decays": max_lr_decays if 'max_lr_decays' in locals() else 100,   
            }   
            logging.info("[Training] Saving final model checkpoint (no validation best available).")   
            save_model(model, optimizer, save_payload, cfg.run_args, hparam_dict)   

    if cfg.run_args.do_test and test_loader is not None:   
        logging.info("[Evaluating] Loading best checkpoint for testing.")   
        checkpoint = torch.load(osp.join(cfg.run_args.save_path, "checkpoint.pt"), map_location=device)   
        model.load_state_dict(checkpoint["model_state_dict"])   
        _saved_margin_flag = bool(getattr(model, "margin_loss_enable", True))
        try:
            if hasattr(model, "margin_loss_enable"):
                model.margin_loss_enable = False
            test_results = test_step(   
                model, test_loader, use_full_ranking=getattr(cfg.run_args, "eval_full_ranking", False)   
            )   
        finally:
            if hasattr(model, "margin_loss_enable"):
                model.margin_loss_enable = _saved_margin_flag
        recalls, NDCGs, MAPs, mrr_res, test_loss, hit_scores = test_results   
        hits_at_1 = hit_scores.get(1, 0.0) if isinstance(hit_scores, dict) else 0.0   
        hits_at_5 = hit_scores.get(5, 0.0) if isinstance(hit_scores, dict) else 0.0   
        hits_at_10 = hit_scores.get(10, 0.0) if isinstance(hit_scores, dict) else 0.0   
        num_params = count_parameters(model)   
        metric_dict = {   
            "hparam/num_params": float(num_params),   
            "hparam/Recall@1": float(recalls.get(1, 0.0)),   
            "hparam/Recall@5": float(recalls.get(5, 0.0)),   
            "hparam/Recall@10": float(recalls.get(10, 0.0)),   
            "hparam/Recall@20": float(recalls.get(20, 0.0)),   
            "hparam/NDCG@1": float(NDCGs.get(1, 0.0)),   
            "hparam/NDCG@5": float(NDCGs.get(5, 0.0)),   
            "hparam/NDCG@10": float(NDCGs.get(10, 0.0)),   
            "hparam/NDCG@20": float(NDCGs.get(20, 0.0)),   
            "hparam/MAP@1": float(MAPs.get(1, 0.0)),   
            "hparam/MAP@5": float(MAPs.get(5, 0.0)),   
            "hparam/MAP@10": float(MAPs.get(10, 0.0)),   
            "hparam/MAP@20": float(MAPs.get(20, 0.0)),   
            "hparam/MRR": float(mrr_res),   
            "hparam/Hits@1": float(hits_at_1),   
            "hparam/Hits@5": float(hits_at_5),   
            "hparam/Hits@10": float(hits_at_10),   
            "hparam/TestLoss": float(test_loss),   
        }   
        logging.info(f"[Evaluating] Test metrics: {metric_dict}")   
        summary_writer.add_hparams(hparam_dict, metric_dict)   

    # Save loss plots (and CSVs) into log_path
    try:
        os.makedirs(cfg.run_args.log_path, exist_ok=True)
        # CSV exports
        if train_step_hist and train_loss_hist:
            with open(osp.join(cfg.run_args.log_path, 'loss_train_step.csv'), 'w') as f:
                f.write('step,loss\n')
                for s, v in zip(train_step_hist, train_loss_hist):
                    f.write(f"{s},{v}\n")
        if valid_step_hist and valid_loss_hist:
            with open(osp.join(cfg.run_args.log_path, 'loss_valid.csv'), 'w') as f:
                f.write('step,loss\n')
                for s, v in zip(valid_step_hist, valid_loss_hist):
                    f.write(f"{s},{v}\n")

        # PNG plots if matplotlib is available
        if _HAS_MPL:
            if train_step_hist and train_loss_hist:
                plt.figure(figsize=(8, 4))
                plt.plot(train_step_hist, train_loss_hist, label='train(step)')
                plt.xlabel('global_step'); plt.ylabel('loss'); plt.title('Train Loss (per step)'); plt.legend(); plt.tight_layout()
                plt.savefig(osp.join(cfg.run_args.log_path, 'loss_train_step.png'))
                plt.close()
            if valid_step_hist and valid_loss_hist:
                plt.figure(figsize=(8, 4))
                plt.plot(valid_step_hist, valid_loss_hist, 'o-', label='valid(eval)')
                plt.xlabel('global_step'); plt.ylabel('loss'); plt.title('Valid Loss (per eval)'); plt.legend(); plt.tight_layout()
                plt.savefig(osp.join(cfg.run_args.log_path, 'loss_valid.png'))
                plt.close()
            if train_step_hist and train_loss_hist:
                plt.figure(figsize=(8, 4))
                plt.plot(train_step_hist, train_loss_hist, label='train(step)')
                if valid_step_hist and valid_loss_hist:
                    plt.plot(valid_step_hist, valid_loss_hist, 'o-', label='valid(eval)')
                plt.xlabel('global_step'); plt.ylabel('loss'); plt.title('Loss Overview'); plt.legend(); plt.tight_layout()
                plt.savefig(osp.join(cfg.run_args.log_path, 'loss_overview.png'))
                plt.close()
    except Exception as e:  # pragma: no cover
        logging.warning(f"[Plot] Failed to save loss artifacts: {e}")

    summary_writer.close()   


if __name__ == "__main__":   
    main()   
