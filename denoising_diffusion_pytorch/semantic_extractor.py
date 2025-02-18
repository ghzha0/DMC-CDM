from FlagEmbedding import BGEM3FlagModel
import pickle
import os
from .cross_attention import Backbone
import json
import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load semantic embedding
class semantic_extractor(nn.Module):
    def __init__(self, dataname, se_emb_path, extractor='bge-m3'):
        super(semantic_extractor, self).__init__()
        self.emb_path = se_emb_path
        self.dataname = dataname
        self.extractor = extractor
        if self.extractor == 'bge-m3':
            model_name = f'checkpoint/{extractor}'
        elif self.extractor == 'llama3':
            model_name = f'../../../share_weight/Meta-Llama-3-8B'
        
        if os.path.exists(self.emb_path):
            self.problem_embedding = self.load_problem_embedding()
        else:
            if self.extractor == 'bge-m3':
                self.model = BGEM3FlagModel(model_name_or_path=model_name)
                self.problem_embedding = self.gen_problem_embedding()
                self.save_problem_embedding(self.problem_embedding)
                self.problem_embedding = torch.FloatTensor(self.problem_embedding)
            elif self.extractor == 'llama3':
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.problem_embedding = self.gen_problem_embedding()
                self.save_problem_embedding(self.problem_embedding)
                self.problem_embedding = torch.FloatTensor(self.problem_embedding)

        # Padding
        problem_embedding = torch.cat([
            self.problem_embedding,
            torch.zeros_like(self.problem_embedding[0]).unsqueeze(0)
        ], dim=0)
        
        self.problem_embedding = nn.Embedding(num_embeddings=problem_embedding.shape[0], embedding_dim=problem_embedding.shape[1])
        self.problem_embedding.weight.data = problem_embedding
        
    def load_problem_embedding(self):
        with open(self.emb_path, mode='rb') as rf:
            embedding = pickle.load(rf)
        return torch.FloatTensor(embedding)

    def save_problem_embedding(self, embedding):
        print(embedding.shape)
        print(f"Saving Embedding at {self.emb_path}!")
        with open(self.emb_path, mode='wb') as wf:
            pickle.dump(embedding, wf)
        print("Saving Done!")
        
    def gen_problem_embedding(self):
        id2prob = {}
        with open(f'./data/{self.dataname}/problem_id_mapping.csv', 'r') as rf:
            for i, row in enumerate(rf.readlines()):
                if i == 0:
                    continue
                else:
                    row = row.split(',')
                    if self.dataname == 'ifly':
                        id2prob[row[0]] = row[1][:-1]
                    else:
                        id2prob[row[1][:-1]] = row[0]

        with open(f'./data/{self.dataname}/processed_problem.json', 'r') as rf:
            problem_info = json.load(rf)

        problems = []
        for id in range(len(id2prob)):
            if 'concepts' in problem_info[id2prob[str(id)]]:
                concept = '主题:' + ','.join(problem_info[id2prob[str(id)]]['concepts'])
            else:
                concept = ''
            content = problem_info[id2prob[str(id)]]['content']
            problems.append(
                concept + content
            )

        if self.extractor == 'bge-m3':
            doc_embeddings = self.model.encode(problems, batch_size=16, max_length=8192, return_dense=True)['dense_vecs']
        elif self.extractor == 'llama3':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            problems = []
            for id in range(len(id2prob)):
                if 'concepts' in problem_info[id2prob[str(id)]]:
                    concept = '主题:' + ','.join(problem_info[id2prob[str(id)]]['concepts'])
                else:
                    concept = ''
                content = problem_info[id2prob[str(id)]]['content']
                problems.append(
                    concept + content
                )

            doc_embeddings = []
            for problem in tqdm.tqdm(problems):
                messages = [
                    {"role": "system", "content": "You are an educational expert who always output the discription of exercises!"}, {"role": "user", "content": problem}
                ]
            
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)

                outputs = self.model.forward(input_ids, output_hidden_states=True)
                doc_embeddings.append(
                    outputs['hidden_states'][-1][:, -1, :].detach().cpu().numpy()
                )

        return doc_embeddings

    def get_problem_embedding(self, log_seqs):
        with torch.no_grad(): # Do not trace the gradient of embedding.
            raw_semantic_embedding = self.problem_embedding(log_seqs)
        return raw_semantic_embedding # (B * N * H)
