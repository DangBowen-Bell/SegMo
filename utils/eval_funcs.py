import os
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from collections import OrderedDict, defaultdict

from utils import new_utils


@torch.no_grad()
def evaluation(
    val_loader, eval_wrapper, model_type, vq_model, 
    text_model=None,
    t2m_trans=None, res_trans=None,
    logger=None, writer=None, epoch=0, best_metrics=None, 
    out_dir=None, 
    repeat_time_mm=1, time_steps=10,
    t2m_tem=1, t2m_cond_scale=2, 
    res_tem=1, res_cond_scale=5):

    # print('t2m_cond_scale:', t2m_cond_scale)
    # print('res_cond_scale:', res_cond_scale)

    def save(file_name, epoch, model):
        state = {
            model_type: model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, file_name)

    device = next(vq_model.parameters()).device
    vq_model.eval()
    if t2m_trans:
        t2m_trans.eval()
    if res_trans:
        res_trans.eval()

    motion_emb_gt_list = []
    motion_emb_pred_list = []
    motion_emb_mm_list = []
    
    R_precision_gt = 0
    R_precision_pred = 0
    matching_score_gt = 0
    matching_score_pred = 0

    def def_list():
        return []

    matching_score_action_gt = defaultdict(def_list, OrderedDict())
    matching_score_action_pred = defaultdict(def_list, OrderedDict())

    nb_sample = 0
    for bi, batch in enumerate(tqdm(val_loader)):
        #* action embedding
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, llm_captions, motion_seg = batch
        motion = motion.to(device)
        m_length = m_length.to(device)

        if text_model is not None:
            caption, action_mask = new_utils.get_action_emb(caption, llm_captions, text_model)

        if model_type == 'vq_model':
            pred_motion, loss_commit, perplexity = vq_model(motion)
        elif model_type == 't2m_trans':
            pred_ids = t2m_trans.generate(
                caption, m_length//4, time_steps, t2m_cond_scale, action_mask, motion_seg, temperature=t2m_tem)
            pred_ids.unsqueeze_(-1)
            pred_motion = vq_model.forward_decoder(pred_ids)
        elif model_type == 'res_trans':
            all_code_ids, all_codes = vq_model.encode(motion)
            if epoch == 0:
                pred_ids = all_code_ids[..., 0:1]
            else:
                pred_ids = res_trans.generate(
                    all_code_ids[..., 0], caption, m_length//4, action_mask, temperature=res_tem, cond_scale=res_cond_scale)
            pred_motion = vq_model.forward_decoder(pred_ids)
        elif model_type == 'all':
            # Only calculate multimodality metric in the final evaluation
            # And use the last generation results to calculate other metrics
            motion_emb_mm_batch_list = []            
            for _ in range(repeat_time_mm):
                pred_ids = t2m_trans.generate(
                    caption, m_length//4, time_steps, t2m_cond_scale, action_mask, motion_seg, temperature=t2m_tem)
                pred_ids = res_trans.generate(
                    pred_ids, caption, m_length//4, action_mask, temperature=res_tem, cond_scale=res_cond_scale)
                pred_motion = vq_model.forward_decoder(pred_ids)
                
                text_emb_pred, motion_emb_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
                motion_emb_mm_batch_list.append(motion_emb_pred.unsqueeze(1))
            motion_emb_mm_batch = torch.cat(motion_emb_mm_batch_list, dim=1)
            motion_emb_mm_list.append(motion_emb_mm_batch)

        text_emb_gt, motion_emb_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        text_emb_pred, motion_emb_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
        
        motion_emb_gt_list.append(motion_emb_gt)
        motion_emb_pred_list.append(motion_emb_pred)

        if motion.shape[0] > 1:
            temp_R, temp_match = calculate_R_precision(text_emb_gt.cpu().numpy(), motion_emb_gt.cpu().numpy(), top_k=3, sum_all=True)
            R_precision_gt += temp_R
            matching_score_gt += temp_match

            temp_R, temp_match = calculate_R_precision(text_emb_pred.cpu().numpy(), motion_emb_pred.cpu().numpy(), top_k=3, sum_all=True)
            R_precision_pred += temp_R
            matching_score_pred += temp_match

            dist_mat_gt = euclidean_distance_matrix(text_emb_gt.cpu().numpy(), motion_emb_gt.cpu().numpy())
            dist_diag_gt = np.diagonal(dist_mat_gt)

            dist_mat_pred = euclidean_distance_matrix(text_emb_pred.cpu().numpy(), motion_emb_pred.cpu().numpy())
            dist_diag_pred = np.diagonal(dist_mat_pred)

            action_num_list = [len(llm_caption) for llm_caption in llm_captions]

            for si in range(motion.shape[0]):
                matching_score_action_gt[str(action_num_list[si])].append(dist_diag_gt[si])
                matching_score_action_pred[str(action_num_list[si])].append(dist_diag_pred[si])
        else:
            # batch size is 1

            temp_R, temp_match = calculate_R_precision(text_emb_gt.cpu().numpy(), motion_emb_gt.cpu().numpy(), top_k=1, sum_all=True)
            R_precision_gt += temp_R
            matching_score_gt += temp_match

            temp_R, temp_match = calculate_R_precision(text_emb_pred.cpu().numpy(), motion_emb_pred.cpu().numpy(), top_k=1, sum_all=True)
            R_precision_pred += temp_R
            matching_score_pred += temp_match

        nb_sample += motion.shape[0]

    motion_emb_gt_np = torch.cat(motion_emb_gt_list, dim=0).cpu().numpy()
    motion_emb_pred_np = torch.cat(motion_emb_pred_list, dim=0).cpu().numpy()
    mu_gt, cov_gt = calculate_activation_statistics(motion_emb_gt_np)
    mu_pred, cov_pred = calculate_activation_statistics(motion_emb_pred_np)

    fid = calculate_frechet_distance(mu_gt, cov_gt, mu_pred, cov_pred)

    R_precision_gt = R_precision_gt / nb_sample
    R_precision_pred = R_precision_pred / nb_sample
    
    matching_score_gt = matching_score_gt / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    diversity_gt = calculate_diversity(motion_emb_gt_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_emb_pred_np, 300 if nb_sample > 300 else 100)

    if model_type == 'all' and repeat_time_mm > 1:
        motion_emb_mm = torch.cat(motion_emb_mm_list, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_emb_mm, 10)
    else:
        multimodality = 0
        
    metrics = {
        'fid': fid,
        'top1': R_precision_pred[0],
        'top2': R_precision_pred[1],
        'top3': R_precision_pred[2],
        'matching': matching_score_pred,
        'div': diversity_pred,
        'mm': multimodality
    }
    
    #! the following code are only executed during training
    if logger:
        msg = f"--> Eval. Epoch {epoch}: " + \
            f"FID. {fid:.4f}, " + \
            f"R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, " + \
            f"matching_score_gt. {matching_score_gt}, Matching_score_pred. {matching_score_pred}, " + \
            f"Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, " + \
            f"MultiModality. {multimodality:.4f}"
        logger.info(msg)

    if writer:
        for name in metrics.keys():
            writer.add_scalar('Test/'+name, metrics[name], epoch)

    if best_metrics:
        small_metric_names = ['fid', 'matching']
        big_metric_names = ['top1', 'top2', 'top3', 'mm']

        for name in best_metrics.keys():
            if (name in small_metric_names and metrics[name] < best_metrics[name]) or \
               (name in big_metric_names and metrics[name] > best_metrics[name]) or \
               (name == 'div' and abs(metrics[name]-diversity_gt) < abs(metrics[name]-diversity_gt)):
                best_metrics[name] = metrics[name]
                if (name == 'fid' or name == 'top1') and out_dir:
                    save_path = os.path.join(out_dir, f'best_{name}.tar')
                    if model_type == 'vq_model':
                        save(save_path, epoch, vq_model)
                    elif model_type == 't2m_trans':
                        save(save_path, epoch, t2m_trans)
                    elif model_type in ['res_trans', 'all']:
                        save(save_path, epoch, res_trans)
        
        return best_metrics

    return metrics


############################################################


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        # print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score_pred = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score_pred
    else:
        return top_k_mat, matching_score_pred
    

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of pred_cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations):
    pred_mu = np.mean(activations, axis=0)
    pred_cov = np.cov(activations, rowvar=False)
    return pred_mu, pred_cov


def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: [num_poses, num_joints, 3]
    pred_joints: [num_poses, num_joints, 3]
    (obtained from recover_from_ric())
    """
    
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # [num_poses, num_joints]
    mpjpe_seq = mpjpe.mean(-1) # [num_poses]

    return mpjpe_seq
