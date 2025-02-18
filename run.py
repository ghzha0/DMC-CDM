import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
import tqdm
import random
import wandb
import yaml
import faiss
from torch.utils.data import DataLoader
import torch.nn.functional as torch_func
from utils import *
from data_io import get_dataset
from denoising_diffusion_pytorch import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(random_seed):
    torch.manual_seed(random_seed)  # cpu
    torch.cuda.manual_seed(random_seed)  # gpu
    np.random.seed(random_seed)  # numpy
    random.seed(random_seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

def pretrain(cfg):
    train_dataset, user_num, item_num = get_dataset(cfg)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=0
    )

    model = inv_cog(user_num, item_num, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr']
    )
    
    best_eval_auc = 0
    patience = cfg['patience']
    for ep in range(cfg['pretrain_epoch']):
        total_encoder_loss = 0.0
        # Get Representation from Collabrative Encoder and Semantic Encoder
        i = 0
        for batch in train_dataloader:
            batch['train_seq'], batch['train_ans'] = batch['train_seq'].to(device), batch['train_ans'].to(device)
            optimizer.zero_grad()
            encoder_loss = model.compute_encoder_loss(batch['train_seq'], batch['train_ans'])
            encoder_loss.backward()
            optimizer.step()
            total_encoder_loss += encoder_loss.item()
            i += 1
    
        print(f'ep: {ep} encoder loss: {total_encoder_loss/i}')
        if ep % 1 == 0:
            print('-' * 100 + 'Eval' + '-' * 100)
            # train_acc, train_auc, train_f1, train_rmse = pre_evalutate(model=model, train_dataloader=train_dataloader, type='train')
            # print(f'ep: {ep} train_acc: {train_acc}, train_auc: {train_auc}, train_f1: {train_f1}, train_rmse: {train_rmse}')
            eval_acc, eval_auc, eval_f1, eval_rmse = pre_evalutate(model=model, train_dataloader=train_dataloader, type='eval')
            
            # log metrics to wandb
            wandb.log(
                {
                    "eval_acc": eval_acc, "eval_auc": eval_auc, "eval_f1": eval_f1, "eval_rmse": eval_rmse
                }
            )

            if eval_auc > best_eval_auc:
                patience = cfg['patience']
                best_eval_auc = eval_auc
                torch.save(model.state_dict(), f'./model/{cfg["dataname"]}_best_pretrained_invcog.pt')
            patience -= 1
            print(f'ep: {ep} eval_acc: {eval_acc}, eval_auc: {eval_auc}, eval_f1: {eval_f1}, eval_rmse: {eval_rmse}')
            print('-' * 100 + 'Eval' + '-' * 100)
        if patience == 0:
            print('Early stop!')
            print('-'*100 + 'Test' + '-'*100)
            break
    
    print('-' * 100 + 'Test' + '-' * 100)
    model.load_state_dict(
        torch.load(f'./model/{cfg["dataname"]}_best_pretrained_invcog.pt', map_location='cuda')
    )
    test_acc, test_auc, test_f1, test_rmse = pre_evalutate(model=model, train_dataloader=train_dataloader, type='test')
    wandb.log(
        {
            'pre_test_acc': test_acc, 'pre_test_auc': test_auc, 'pre_test_f1': test_f1, 'pre_test_rmse': test_rmse
        }
    )
    print('Test@config:')
    print(cfg)
    print(f'pre_test_acc: {test_acc}, pre_test_auc: {test_auc}, pre_test_f1: {test_f1}, pre_test_rmse: {test_rmse}')
    
def train(cfg):
    train_dataset, user_num, item_num = get_dataset(cfg)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    ### Build Gaussian Diffusion ###
    if cfg['mean_type'] == 'x0':
        mean_type = ModelMeanType.START_X
    elif cfg['mean_type'] == 'eps':
        mean_type = ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % cfg['mean_type'])
    model = inv_cog(user_num, item_num, cfg)
    diffusion = GaussianDiffusion(
        model=model,
        device=device,
        mean_type=mean_type,
        noise_schedule=cfg['noise_schedule'],
        noise_scale=cfg['noise_scale'],
        noise_min=cfg['noise_min'],
        noise_max=cfg['noise_max'],
        steps=5
    ).to(device)
    diffusion.model.load_state_dict(torch.load(f'./model/{cfg["dataname"]}_best_pretrained_invcog.pt', map_location=device))
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=cfg['lr'])

    best_eval_auc = 0
    patience = cfg['patience']

    # Freeze encoder parameters
    for name, param in diffusion.model.named_parameters():
        if 'cross_att_1' in name:
            param.requires_grad = False
        if 'cross_att_2' in name:
            param.requires_grad = False
        if 'backbone' in name:
            param.requires_grad = False
        if 'adapter' in name:
            param.requires_grad = False
        if 'ce' in name:
            param.requires_grad = False
        if 'se' in name:
            param.requires_grad = False

    # Build User Pool
    with torch.no_grad():
        user_pool = []
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, desc='Build User Pool')):
            batch['train_seq'], batch['train_ans'] = batch['train_seq'].to(device), batch['train_ans'].to(device)
            user_pool.append(
                diffusion.model.get_user_embedding(batch['train_seq'], batch['train_ans']).detach().cpu().numpy()
            )
        user_pool = np.concatenate(user_pool, axis=0)
        index = faiss.IndexFlatIP(1024)
        index.add(user_pool/np.linalg.norm(user_pool, axis=1, keepdims=True))

    for ep in range(cfg['epoch']):
        total_reconstruction_loss = 0.0
        total_diffusion_loss = 0.0
        total_encoder_loss = 0.0
        start_index = 0
        # Train Diffusion Aggregator
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, desc="Training!")):
            batch['train_seq'], batch['train_ans'] = batch['train_seq'].to(device), batch['train_ans'].to(device)
            batch['eval_seq'], batch['eval_ans'] = batch['eval_seq'].to(device), batch['eval_ans'].to(device)
            
            with torch.no_grad():
                user_embedding = diffusion.model.get_user_embedding(batch['train_seq'], batch['train_ans']).detach().cpu().numpy()
                _, I = index.search(user_embedding/np.linalg.norm(user_embedding, axis=1, keepdims=True), cfg['k'])
            
            input = {
                'tgt_seq_logs': [],
                'tgt_answers': [],
                'ctx_seq_logs': [],
                'ctx_answers': []
            }

            assert (I[:, 0] == start_index + np.arange(len(I))).any(), "Exist the first one is not the same user!"
            start_index += cfg['batch_size']

            for top_k in range(cfg['k']):
                similar_user = I[:, top_k]
                input['tgt_seq_logs'].append(batch['train_seq'])
                input['tgt_answers'].append(batch['train_ans'])
                input['ctx_seq_logs'].append(torch.stack([train_dataset[i]['train_seq'] for i in similar_user], dim=0).to(device))
                input['ctx_answers'].append(torch.stack([train_dataset[i]['train_ans'] for i in similar_user], dim=0).to(device))

            input['tgt_seq_logs'] = torch.cat(input['tgt_seq_logs'], dim=0) # [k * B, N]
            input['tgt_answers'] = torch.cat(input['tgt_answers'], dim=0) # [k * B, N]
            input['ctx_seq_logs'] = torch.cat(input['ctx_seq_logs'], dim=0) # [k * B, N]
            input['ctx_answers'] = torch.cat(input['ctx_answers'], dim=0) # [k * B, N]
            terms = diffusion.train_losses(input)
            diffusion_loss = terms['loss'].mean()

            # reconstruction loss after diffusion
            x_pred = terms['model_output']
            reconstruction_loss1 = diffusion.model.compute_reconstruction_loss(x_pred, input['tgt_seq_logs'], input['tgt_answers'])
            reconstruction_loss2 = diffusion.model.compute_reconstruction_loss(x_pred, input['ctx_seq_logs'], input['ctx_answers'])
            reconstruction_loss = reconstruction_loss1 + reconstruction_loss2
            loss = diffusion_loss + reconstruction_loss
            loss.backward()

            # gradient clip
            nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=100.0)
            optimizer.step()
            total_diffusion_loss += diffusion_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
        print(f'ep: {ep} encoder loss:{total_encoder_loss/i} diffusion loss: {total_diffusion_loss/(i * cfg["k"])} reconstruction loss: {total_reconstruction_loss/(i * cfg["k"])}')
        # Compute train metric
        # train_acc, train_auc, train_f1, train_rmse = test(diffusion=diffusion, train_dataloader=train_dataloader, train_dataset=train_dataset, index=index, type='train', k=cfg['k'], p_steps=cfg['p_steps'])
        # print(f'ep: {ep} train_acc: {train_acc}, train_auc: {train_auc}, train_f1: {train_f1}, train_rmse: {train_rmse}')

        # Evaluate
        if ep % 1 == 0:
            print('-' * 50 + 'Eval' + '-' * 50)
            eval_acc, eval_auc, eval_f1, eval_rmse = test(diffusion=diffusion, train_dataloader=train_dataloader, train_dataset=train_dataset, index=index, type='eval', k=cfg['k'], p_steps=cfg['p_steps'])   
            # log metrics to wandb
            wandb.log(
                {
                    "eval_acc": eval_acc, "eval_auc": eval_auc, "eval_f1": eval_f1, "eval_rmse": eval_rmse
                }
            )
            if eval_auc > best_eval_auc:
                patience = cfg['patience']
                best_eval_auc = eval_auc
                torch.save(diffusion.model.state_dict(), f'./model/{cfg["dataname"]}_best_diffused_invcog.pt')
            patience -= 1
            print(f'ep: {ep} eval_acc: {eval_acc}, eval_auc: {eval_auc}, eval_f1: {eval_f1}, eval_rmse: {eval_rmse}')
            print('-' * 50 + 'Eval' + '-' * 50)
        if patience == 0:
            print('Early stop!')
            break
        
    # Test
    print('-' * 50 + 'Test' + '-' * 50)
    diffusion.model.load_state_dict(
        torch.load(f'./model/{cfg["dataname"]}_best_diffused_invcog.pt', map_location='cuda')
    )
    test_acc, test_auc, test_f1, test_rmse = test(diffusion=diffusion, train_dataloader=train_dataloader, train_dataset=train_dataset, index=index, type='test', k=cfg['k'], p_steps=cfg['p_steps'])   
    wandb.log(
        {
            'test_acc': test_acc, 'test_auc': test_auc, 'test_f1': test_f1, 'test_rmse': test_rmse
        }
    )
    print('Test@config:')
    print(cfg)
    print(f'test_acc: {test_acc}, test_auc: {test_auc}, test_f1: {test_f1}, test_rmse: {test_rmse}')

def pre_evalutate(model, train_dataloader, type='eval'):
    model.eval()
    with torch.no_grad():
        predictedIndices = []
        GroundTruth = []
        for batch in train_dataloader:
            batch['train_seq'], batch['train_ans'] = batch['train_seq'].to(device), batch['train_ans'].to(device)
            if type =='train':
                eval_seq = batch['train_seq'].to(device)
                eval_ans = batch['train_ans'].to(device)
            elif type == 'eval':
                eval_seq = batch['eval_seq'].to(device)
                eval_ans = batch['eval_ans'].to(device)
            elif type == 'test':
                eval_seq = batch['test_seq'].to(device)
                eval_ans = batch['test_ans'].to(device)
            mask = (eval_ans != 0)

            x_pred = model.get_user_embedding(batch['train_seq'], batch['train_ans']).unsqueeze(1)
            # Compute logits
            with torch.no_grad():
                prob_embs = model._get_embedding(eval_seq)
            prob_logits = torch.sigmoid(
                model.predictor(x_pred - prob_embs)
            ).squeeze(-1)
            predictedIndices.extend(torch.masked_select(prob_logits, mask).detach().cpu().tolist())
            GroundTruth.extend(torch.masked_select(eval_ans, mask).detach().cpu().tolist())

        predictedIndices = np.array(predictedIndices)
        GroundTruth = np.array(GroundTruth)
        GroundTruth = (GroundTruth > 0).astype(int)
        return compute_metrics(y_true=GroundTruth, y_pred=predictedIndices)

def test(diffusion, train_dataloader, train_dataset, index, type='eval', k=2, p_steps=2):
    diffusion.model.eval()
    with torch.no_grad():
        predictedIndices = []
        GroundTruth = []
        start_index = 0
        for batch in tqdm.tqdm(train_dataloader, desc='Evaluating!'):
            batch['train_seq'], batch['train_ans'] = batch['train_seq'].to(device), batch['train_ans'].to(device)
            batch['eval_seq'], batch['eval_ans'] = batch['eval_seq'].to(device), batch['eval_ans'].to(device)
            batch['test_seq'], batch['test_ans'] = batch['test_seq'].to(device), batch['test_ans'].to(device)
            if type =='train':
                eval_seq = batch['train_seq'].to(device)
                eval_ans = batch['train_ans'].to(device)
            elif type == 'eval':
                eval_seq = batch['eval_seq'].to(device)
                eval_ans = batch['eval_ans'].to(device)
            elif type == 'test':
                eval_seq = batch['test_seq'].to(device)
                eval_ans = batch['test_ans'].to(device)
            mask = (eval_ans != 0)

            with torch.no_grad():
                user_embedding = diffusion.model.get_user_embedding(batch['train_seq'], batch['train_ans']).detach().cpu().numpy()
            B, N = eval_seq.shape[0], eval_seq.shape[1]
            D, I = index.search(user_embedding/np.linalg.norm(user_embedding,axis=1,keepdims=True), k)

            assert (I[:, 0] == start_index + np.arange(len(I))).any(), "Exist the first one is not the same user!"
            start_index += B

            input = {
                'tgt_seq_logs': [],
                'tgt_answers': [],
                'ctx_seq_logs': [],
                'ctx_answers': []
            }

            for top_k in range(k):
                similar_user = I[:, top_k]
                input['tgt_seq_logs'].append(batch['train_seq'])
                input['tgt_answers'].append(batch['train_ans'])
                input['ctx_seq_logs'].append(torch.stack([train_dataset[i]['train_seq'] for i in similar_user], dim=0).to(device))
                input['ctx_answers'].append(torch.stack([train_dataset[i]['train_ans'] for i in similar_user], dim=0).to(device))

            input['tgt_seq_logs'] = torch.cat(input['tgt_seq_logs'], dim=0) # [k * B, N]
            input['tgt_answers'] = torch.cat(input['tgt_answers'], dim=0) # [k * B, N]
            input['ctx_seq_logs'] = torch.cat(input['ctx_seq_logs'], dim=0) # [k * B, N]
            input['ctx_answers'] = torch.cat(input['ctx_answers'], dim=0) # [k * B, N]

            x_pred = diffusion.p_sample(input, p_steps).unsqueeze(1) # [k * B, 1, H]
            # x_pred = diffusion(input, p_steps).unsqueeze(1) # [k * B, 1, H]
            
            with torch.no_grad():
                prob_embs = diffusion.model._get_embedding(eval_seq) # [B, N, H]
                prob_embs = prob_embs.repeat(k, 1, 1) # [k * B, N, H]          
            k_prob_logits = torch.sigmoid(
                diffusion.model.predictor(x_pred - prob_embs)
            ).squeeze(-1) # [k * B, N]
            k_prob_logits = k_prob_logits.view(k, B, N).permute(1, 2, 0) # [B, N, k]
            
            # Majority-Vote
            D = torch.FloatTensor(D).to(device) # Normalize for distance
            D = torch_func.normalize(D, p=1, dim=1).unsqueeze(1) # [B * 1 * K]
            prob_logits = torch.sum(k_prob_logits * D, dim=-1) / torch.sum(D, dim=-1)
            predictedIndices.extend(torch.masked_select(prob_logits, mask).detach().cpu().tolist())
            GroundTruth.extend(torch.masked_select(eval_ans, mask).detach().cpu().tolist())
    
    predictedIndices = np.array(predictedIndices)
    GroundTruth = np.array(GroundTruth)
    GroundTruth = (GroundTruth > 0).astype(int)
    return compute_metrics(y_true=GroundTruth, y_pred=predictedIndices)


if __name__ == "__main__":
    with open('./train_config/model_config.yaml') as rF:
        cfg = yaml.safe_load(rF)['invcog']
    cfg['se_emb_path'] = f"./embedding/{cfg['dataname']}/{cfg['semantic_extractor']}/item_embedding.pkl"    
    print(cfg)
    set_seed(cfg['seed'])
    # wandb.init(
    #     project="invcog-pretrain",
    #     config=cfg,
    #     name=f"{cfg['dataname']}_{cfg['semantic_extractor']}_{cfg['k']}"
    # )

    if cfg['pretrain']:
        pretrain(cfg)
    if cfg['train'] and not cfg['wodiff']:
        train(cfg)
