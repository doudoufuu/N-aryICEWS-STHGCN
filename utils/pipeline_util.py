import json  # [RESTORED]
import logging  # [RESTORED]
import os.path as osp  # [RESTORED]
import random  # [ADDED-DEBUG]

import numpy as np  # [RESTORED]
import torch  # [RESTORED]
from tqdm import tqdm  # [RESTORED]

from metric import (  # [RESTORED]
    map_k,  # [RESTORED]
    mrr,  # [RESTORED]
    ndcg,  # [RESTORED]
    recall,  # [RESTORED]
)  # [RESTORED]


def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):  # [RESTORED]
    """Persist model/optimizer states together with auxiliary training variables."""  # [RESTORED]
    with open(osp.join(run_args.log_path, 'config.json'), 'w') as fjson:  # [RESTORED]
        for key, value in list(argparse_dict.items()):  # [RESTORED]
            if isinstance(value, torch.Tensor):  # [RESTORED]
                argparse_dict[key] = value.cpu().numpy().tolist()  # [RESTORED]
        json.dump(argparse_dict, fjson)  # [RESTORED]

    torch.save({  # [RESTORED]
        **save_variable_list,  # [RESTORED]
        'model_state_dict': model.state_dict(),  # [RESTORED]
        'optimizer_state_dict': optimizer.state_dict()  # [RESTORED]
    }, osp.join(run_args.save_path, 'checkpoint.pt'))  # [RESTORED]


def count_parameters(model):  # [RESTORED]
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # [RESTORED]


def test_step(model, data_loader, ks=(1, 5, 10, 20), use_full_ranking=False):  # [MODIFIED-TRAIN]
    """Evaluate the model on the given loader and return ranking metrics."""  # [ADDED-TRAIN]
    model.eval()  # [ADDED-TRAIN]
    loss_list = []  # [ADDED-TRAIN]
    pred_list = []  # [ADDED-TRAIN]
    label_list = []  # [ADDED-TRAIN]

    preview_batches = 0  # [ADDED-DEBUG]
    preview_batch_limit = 3  # [ADDED-DEBUG]
    debug_sample_limit = 5  # [ADDED-DEBUG]
    debug_sample_logs = 0  # [ADDED-DEBUG]
    label_pos_check_done = False  # [ADDED-DEBUG]
    with torch.no_grad():  # [ADDED-TRAIN]
        for batch in tqdm(data_loader):  # [ADDED-TRAIN]
            batch = batch.to(model.device)  # [ADDED-TRAIN]
            output = model(batch)  # [ADDED-TRAIN]
            loss_list.append(output["loss"].detach().cpu().numpy().tolist())  # [ADDED-TRAIN]

            if use_full_ranking and "logits" in output:  # [ADDED-TRAIN]
                rankings = output["logits"].argsort(dim=1, descending=True)  # [ADDED-TRAIN]
                pred_list.append(rankings.detach().cpu())  # [ADDED-TRAIN]
                debug_idx = random.randrange(rankings.size(0)) if rankings.size(0) > 0 else 0  # [ADDED-DEBUG]
                logits_sample = output["logits"][debug_idx].detach().cpu()  # [ADDED-DEBUG]
                top_scores, top_indices = logits_sample.topk(min(5, logits_sample.size(0)))  # [ADDED-DEBUG]
                label_idx_sample = int(batch.y[debug_idx].item()) if hasattr(batch, "y") else None  # [ADDED-DEBUG]
                label_score_sample = float(logits_sample[label_idx_sample].item()) if label_idx_sample is not None else float("nan")  # [ADDED-DEBUG]
                if debug_sample_logs < debug_sample_limit:  # [ADDED-DEBUG]
                    logging.debug(  # [ADDED-DEBUG]
                        "[Evaluating] Full-ranking sample idx=%d label_idx=%s label_score=%.6f top_pairs=%s",  # [ADDED-DEBUG]
                        debug_idx,  # [ADDED-DEBUG]
                        label_idx_sample,  # [ADDED-DEBUG]
                        label_score_sample,  # [ADDED-DEBUG]
                        [(int(idx), float(score)) for idx, score in zip(top_indices.tolist(), top_scores.tolist())],  # [ADDED-DEBUG]
                    )  # [ADDED-DEBUG]
                    debug_sample_logs += 1  # [ADDED-DEBUG]
            else:  # [ADDED-TRAIN]
                # One-time sanity check: ensure candidates' first column equals labels
                if not label_pos_check_done:
                    try:
                        c0 = batch.candidates[:, 0]
                        mism = (c0 != batch.y)
                        if mism.any():
                            idxs = torch.nonzero(mism, as_tuple=False).view(-1)[:5].cpu().tolist()
                            examples = [
                                (
                                    int(batch.candidates[i, 0].item()),
                                    int(batch.y[i].item()),
                                    batch.candidates[i].detach().cpu().tolist(),
                                )
                                for i in idxs
                            ]
                            logging.error(
                                "[Evaluating] Label not at column 0 in %d/%d rows. Examples (cand0,label,row): %s",
                                int(mism.sum().item()),
                                int(mism.numel()),
                                examples,
                            )
                        else:
                            logging.debug("[Evaluating] Sanity check passed: candidates[:,0] matches labels on first batch.")
                    except Exception as e:
                        logging.warning("[Evaluating] Failed to verify label position: %s", e)
                    label_pos_check_done = True
                scores = torch.cat([output["gold_scores"], output["negative_scores"]], dim=1)  # [ADDED-TRAIN]
                rankings = scores.argsort(dim=1, descending=True)  # [ADDED-TRAIN]
                candidates = batch.candidates.to(device=scores.device)  # [ADDED-TRAIN]
                sorted_candidates = candidates.gather(1, rankings.to(candidates.device))  # [ADDED-TRAIN]                
                pred_list.append(sorted_candidates)  # [ADDED-TRAIN]
                if scores.size(0) > 0:  # [ADDED-DEBUG]
                    debug_idx = random.randrange(scores.size(0))  # [ADDED-DEBUG]
                    cand_ids = batch.candidates[debug_idx].detach().cpu().tolist()  # [ADDED-DEBUG]
                    cand_scores = scores[debug_idx].detach().cpu().tolist()  # [ADDED-DEBUG]
                    label_idx_sample = int(batch.y[debug_idx].item())  # [ADDED-DEBUG]
                    candidate_pairs = list(zip(cand_ids, cand_scores))  # [ADDED-DEBUG]
                    sorted_pairs = sorted(candidate_pairs, key=lambda x: x[1], reverse=True)  # [ADDED-DEBUG]
                    label_score_sample = next((score for cid, score in candidate_pairs if cid == label_idx_sample), float("nan"))  # [ADDED-DEBUG]
                    top_gap = float(sorted_pairs[0][1] - sorted_pairs[1][1]) if len(sorted_pairs) > 1 else 0.0  # [ADDED-DEBUG]
                    if debug_sample_logs < debug_sample_limit:  # [ADDED-DEBUG]
                        logging.debug(  # [ADDED-DEBUG]
                            "[Evaluating] Candidate sample idx=%d label_idx=%d label_score=%.6f top_gap=%.6f ranking=%s",  # [ADDED-DEBUG]
                            debug_idx,  # [ADDED-DEBUG]
                            label_idx_sample,  # [ADDED-DEBUG]
                            label_score_sample,  # [ADDED-DEBUG]
                            top_gap,  # [ADDED-DEBUG]
                            [(int(cid), float(score)) for cid, score in sorted_pairs],  # [ADDED-DEBUG]
                        )  # [ADDED-DEBUG]
                        debug_sample_logs += 1  # [ADDED-DEBUG]
                if preview_batches < preview_batch_limit and debug_sample_logs < debug_sample_limit:  # [ADDED-DEBUG]
                    cand_preview = batch.candidates[:5].detach().cpu()  # [ADDED-DEBUG]
                    label_preview = batch.y[:5].detach().cpu() if hasattr(batch, "y") else None  # [ADDED-DEBUG]
                    logging.debug("[Evaluating] Candidate set preview (first 5 rows): %s", cand_preview.tolist())  # [ADDED-DEBUG]
                    if label_preview is not None:  # [ADDED-DEBUG]
                        logging.debug("[Evaluating] Label preview (first 5): %s", label_preview.tolist())  # [ADDED-DEBUG]
                    preview_batches += 1  # [ADDED-DEBUG]
                    debug_sample_logs += 1  # [ADDED-DEBUG]

            label_list.append(batch.y.view(-1, 1).detach().cpu())  # [ADDED-TRAIN]

    if not pred_list:  # [ADDED-TRAIN]
        logging.warning("[Evaluating] No evaluation batches were processed.")  # [ADDED-TRAIN]
        empty_hits = {k: 0.0 for k in (1, 5, 10)}  # [ADDED-METRIC]
        return {}, {}, {}, 0.0, 0.0, empty_hits  # [ADDED-TRAIN]

    pred_ = torch.cat(pred_list, dim=0).cpu()  # [ADDED-TRAIN]
    label_ = torch.cat(label_list, dim=0).cpu()  # [ADDED-TRAIN]

    recalls, NDCGs, MAPs = {}, {}, {}  # [ADDED-TRAIN]
    avg_loss = float(np.mean(loss_list)) if loss_list else 0.0  # [ADDED-TRAIN]
    logging.info(f"[Evaluating] Average loss: {avg_loss}")  # [ADDED-TRAIN]

    label_flat = label_.view(-1, 1)  # [ADDED-METRIC]
    max_rank = pred_.size(1)  # [ADDED-METRIC]
    hit_ks = (1, 5, 10)  # [ADDED-METRIC]
    hit_scores = {}  # [ADDED-METRIC]
    for hit_k in hit_ks:  # [ADDED-METRIC]
        effective_k = min(hit_k, max_rank)  # [ADDED-METRIC]
        if effective_k <= 0:  # [ADDED-METRIC]
            hit_scores[hit_k] = 0.0  # [ADDED-METRIC]
            logging.info(f"[Evaluating] Hits@{hit_k} : {hit_scores[hit_k]}")  # [ADDED-METRIC]
            continue  # [ADDED-METRIC]
        in_topk = (pred_[:, :effective_k] == label_flat).any(dim=1).float().mean().item()  # [ADDED-METRIC]
        hit_scores[hit_k] = float(in_topk)  # [ADDED-METRIC]
        logging.info(f"[Evaluating] Hits@{hit_k} : {hit_scores[hit_k]}")  # [ADDED-METRIC]
    hits_at_1 = hit_scores.get(1, 0.0)  # [ADDED-METRIC]

    for k_ in ks:  # [ADDED-TRAIN]
        effective_k = min(k_, pred_.size(1))  # [ADDED-TRAIN]
        if effective_k <= 0:  # [ADDED-TRAIN]
            recalls[k_] = 0.0  # [ADDED-TRAIN]
            NDCGs[k_] = 0.0  # [ADDED-TRAIN]
            MAPs[k_] = 0.0  # [ADDED-TRAIN]
            continue  # [ADDED-TRAIN]
        recalls[k_] = recall(label_, pred_, effective_k).cpu().detach().numpy().tolist()  # [ADDED-TRAIN]
        NDCGs[k_] = ndcg(label_, pred_, effective_k).cpu().detach().numpy().tolist()  # [ADDED-TRAIN]
        MAPs[k_] = map_k(label_, pred_, effective_k).cpu().detach().numpy().tolist()  # [ADDED-TRAIN]
        logging.info(  # [ADDED-TRAIN]
            f"[Evaluating] Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}"  # [ADDED-TRAIN]
        )  # [ADDED-TRAIN]

    mrr_res = mrr(label_, pred_).cpu().detach().numpy().tolist()  # [ADDED-TRAIN]
    logging.info(f"[Evaluating] MRR : {mrr_res}")  # [ADDED-TRAIN]
    return recalls, NDCGs, MAPs, mrr_res, avg_loss, hit_scores  # [ADDED-TRAIN]




