import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from conch.open_clip_custom import tokenize, get_tokenizer
from conch.downstream.utils import AverageMeter, merge_dict
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, 
                             classification_report, roc_auc_score)
from tqdm import tqdm

@torch.no_grad()
def zero_shot_classifier(model, classnames, templates, tokenizer=None, device=None):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    zeroshot_weights = []
    for classnames_for_class in classnames:
        embeddings_for_class = []
        for classname in classnames_for_class:
            texts = [template.replace('CLASSNAME', classname) for template in templates]
            token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
            token_ids = token_ids.to(device)
            classname_embeddings = model.encode_text(token_ids)
            # classname_embeddings: [num_templates, embedding_dim]
            embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))

        # class_embedding: [num_classnames, num_templates, embedding_dim]
        class_embedding = torch.stack(embeddings_for_class, dim=0)
        # over all templates and classnames
        class_embedding = class_embedding.mean(dim=(0, 1))
        class_embedding /= class_embedding.norm()

        # class_embedding: [embedding_dim]
        zeroshot_weights.append(class_embedding)

    # zeroshot_weights: [embedding_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def topj_pooling(logits, topj):
    """
    logits: N x C logit for each patch
    topj: tuple of the top number of patches to use for pooling
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, _ = logits.topk(maxj, 0, True, True) # maxj x C
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    pooled_logits = {key: val for key,val in preds.items()}    
    preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
    return preds, pooled_logits

@torch.no_grad()
def run_mizero(model, classifier, dataloader, device, topj = (1,5,10,50,100), 
        dump_results = False, dump_patch_level = False, 
        metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1']):                                            
        
    dict_keys = list(topj)
    meters = {j: AverageMeter() for j in dict_keys}

    logits_all, targets_all, patch_logits_all, coords_all, preds_all = {}, [], [], [], {}
    for idx, data in enumerate(tqdm(dataloader)): # batch size is always 1, 
        image_features = data['img'].to(device).squeeze(0)
        target = data['label'].to(device)
        coords = data['coords']
        
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        coords_all.append(coords)

        image_features = model.visual.forward_project(image_features)            
        image_features = F.normalize(image_features, dim=-1) 
        logits = image_features @ classifier
        
        if dump_results and dump_patch_level:
            patch_logits_all.append(logits.cpu().numpy())

        preds, pooled_logits = topj_pooling(logits,
                                        topj = topj)
        results = {key: (val == target).float().item() for key, val in preds.items()}
        
        preds_all = merge_dict(preds_all, preds, value_fn = lambda x: x.item())
        logits_all = merge_dict(logits_all, pooled_logits, value_fn = lambda x: x.cpu().numpy())
        targets_all.append(target.cpu().numpy())

        for j in topj:
            meters[j].update(results[j], n=1) # Update AverageMeters with new results

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = {key: np.concatenate(logits_all[key], axis=0) for key in dict_keys}
    probs_all = {key: F.softmax(torch.from_numpy(logits_all[key]) * model.logit_scale.exp().item(), dim=1).numpy() for key in dict_keys}
    # Compute metrics
    preds_all = {key: np.array(preds_all[key]) for key in dict_keys}
    baccs = {key: balanced_accuracy_score(targets_all, val) for key, val in preds_all.items()}
    cls_rep = {key: classification_report(targets_all, val, output_dict=True, zero_division=0) for key, val in preds_all.items()}
    kappas = {key: cohen_kappa_score(targets_all, val) for key, val in preds_all.items()}
    weighted_kappas = {key: cohen_kappa_score(targets_all, val, weights='quadratic') for key, val in preds_all.items()}
    roc_aucs = {}
    for key, probs in probs_all.items():
        n_classes = probs.shape[1]
        if n_classes == 2:
            class_probs = probs[:,1]
            roc_kwargs = {}
        else:
            class_probs = probs
            roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}        
        try:
            roc_auc = roc_auc_score(targets_all, class_probs, **roc_kwargs)
        except ValueError:
            roc_auc = np.nan
        roc_aucs[key] = roc_auc

    # Get final accuracy across all images
    accs = {j: meters[j].avg for j in topj}

    dump = {}
    results = {'acc': accs, 
            'bacc': baccs, 
            'report': cls_rep, 
            'kappa': kappas,
            'weighted_kappa': weighted_kappas, # quadratic weights
            'roc_auc': roc_aucs,
            'weighted_f1': {key: cls_rep[key]['weighted avg']['f1-score'] for key in dict_keys}}
    results = {k: results[k] for k in metrics}
    if dump_results:
        # dump slide level predictions
        dump['logits'] = logits_all
        dump['targets'] = targets_all
        dump['preds'] = preds_all
        if hasattr(model, "logit_scale"):
            dump['temp_scale'] = model.logit_scale.exp().item()
        
        # dump patch level predictions + coordinates
        if dump_patch_level: 
            dump['patch_logits'] = patch_logits_all
            dump['coords'] = coords_all

    return results, dump

def dataloding_post_process(batch):
    if not isinstance(batch, dict):
        return {'img': batch[0], 'label': batch[1]}
    return batch

@torch.no_grad()
def run_zeroshot(model, classifier, dataloader, device, dump_results=False, 
        metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1']):
    acc_meter = AverageMeter() 

    logits_all, targets_all, preds_all = [], [], []
    for batch_idx, data in enumerate(tqdm(dataloader)): 
        data = dataloding_post_process(data)
        imgs = data['img'].to(device)
        targets = data['label'].to(device)
        image_features = model.encode_image(imgs)
        batch_size = targets.size(0)

        logits = image_features @ classifier
        preds = logits.argmax(dim=1)

        logits_all.append(logits.cpu().numpy())
        targets_all.append(targets.cpu().numpy())
        preds_all.append(preds.cpu().numpy())

        acc_meter.update((preds == targets).float().mean().item(), n=batch_size) # Update AverageMeters with new results

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = F.softmax(torch.from_numpy(logits_all) * model.logit_scale.exp().item(), dim=1).numpy()
    preds_all = np.concatenate(preds_all, axis=0)
    bacc = balanced_accuracy_score(targets_all, preds_all)
    weighted_kappa = cohen_kappa_score(targets_all, preds_all, weights='quadratic')
    kappa = cohen_kappa_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0) 
    acc = acc_meter.avg

    n_classes = probs_all.shape[1]
    if n_classes == 2:
        class_probs = probs_all[:,1]
        roc_kwargs = {}
    else:
        class_probs = probs_all
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}        
    try:
        roc_auc = roc_auc_score(targets_all, class_probs, **roc_kwargs)
    except ValueError:
        roc_auc = np.nan
    
    dump = {}
    results = {'acc': acc, 
            'bacc': bacc, 
            'weighted_kappa': weighted_kappa,
            'kappa': kappa,
            'roc_auc': roc_auc,
            'weighted_f1': cls_rep['weighted avg']['f1-score']}
    results = {k: results[k] for k in metrics}
    
    if dump_results:
        dump['logits'] = logits_all
        dump['targets'] = targets_all
        dump['preds'] = preds_all
        if hasattr(model, "logit_scale"):
            dump['temp_scale'] = model.logit_scale.exp().item()
    return results, dump