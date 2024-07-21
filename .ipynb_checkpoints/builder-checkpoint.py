import string
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F


# Замените пути до директорий и файлов! Можете использовать для локальной отладки.
# При проверке на сервере пути будут изменены
glue_qqp_dir = './data/QQP'
glove_path = './data/glove.6B.50d.txt'

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mu)**2) / ((self.sigma) ** 2))

class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        def generate_mu (K:int = 5) -> List[float]:
            step = 1 / (K - 1)
            return [ i * step - 1 for i in range(2 * K - 1) if i % 2 != 0] + [1]
        mu_s = generate_mu(self.kernel_num)
        for mu in mu_s[:-1]:
          kernel = GaussianKernel(mu = mu, sigma=self.sigma)
          kernels.append(kernel)
        kernel_last = GaussianKernel(mu = mu_s[-1], sigma=self.exact_sigma)
        return kernels.append(kernel_last)

    def _get_mlp(self) -> torch.nn.Sequential:
        layers = torch.nn.Sequential()
        if len(self.out_layers) <= 1 :
          last_size = self.kernel_num
        else:
          for index, value in enumerate(self.out_layers):
            if index+1 < len(self.out_layers):
              layers.append(torch.nn.Linear(value, self.out_layers[index+1]))
              layers.append(torch.nn.ReLU())
              last_size = self.out_layers[index+1]
        layers.append(torch.nn.Linear(last_size, 1))
        return layers

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        embed_query = self.embeddings(query.long())
        embed_doc = self.embeddings(doc.long())

        matching_matrix = torch.einsum(
                          'jib, jmb -> jim',
                          F.normalize(embed_query, p = 2, dim = -1),
                          F.normalize(embed_doc, p = 2, dim = -1)
                          )

        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        # допишите ваш код здесь
        return [self.vocab.get(word, self.oov_val) for word in tokenized_text]


    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        # допишите ваш код здесь
        text = self.preproc_func(self.idx_to_text_mapping[idx])
        return self._tokenized_text_to_index(text)

    def __getitem__(self, idx:int):
        pass

class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        cur_row = self.index_pairs_or_triplets[idx]
        left_idxs = self._convert_text_idx_to_token_idxs(cur_row[0])[:self.max_len]
        r1_idxs = self._convert_text_idx_to_token_idxs(cur_row[1])[:self.max_len]
        r2_idxs = self._convert_text_idx_to_token_idxs(cur_row[2])[:self.max_len]
        pair1 = {'query': left_idxs, 'document':r1_idxs}
        pair2 = {'query': left_idxs, 'document':r2_idxs}
        target = cur_row[3]
        return (pair1, pair2, target)

class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        cur_row = self.index_pairs_or_triplets[idx]
        left_idxs = self._convert_text_idx_to_token_idxs(cur_row[0][:self.max_len])
        r1_idxs = self._convert_text_idx_to_token_idxs(cur_row[1][:self.max_len])
        pair1 = {'query': left_idxs, 'document':r1_idxs}
        target = cur_row[2]
        return (pair1, target)

def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels

class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10,
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
             [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep
        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
              self.idx_to_text_mapping_dev,
              vocab=self.vocab, oov_val=self.vocab['OOV'],
              preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        if pd.__version__ > '1.3.0':
          glue_df = pd.read_csv(
              self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', on_bad_lines='skip', dtype=object)
        else:
          glue_df = pd.read_csv(
              self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def handle_punctuation(self, inp_str: str) -> str:
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return  inp_str.translate(translation_table)

    def simple_preproc(self, inp_str: str) -> List[str]:
        # допишите ваш код здесь
        inp_str = self.handle_punctuation(inp_str.lower())
        return nltk.word_tokenize(inp_str)

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        # допишите ваш код здесь
        arr_dict = np.array(list(vocab.items()))
        mask = arr_dict[:,1].astype(int) > min_occurancies
        return {row[0]: int(row[1]) for row in arr_dict[mask,]}

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        # допишите ваш код здесь
        arr = np.array([], dtype = str)
        for df in list_of_df:
          for col in df:
            if (df[col].dtype == object) & (col in ['text_left', 'text_right']):
              arr_current = df[col].to_numpy().reshape(-1)
              arr = np.concatenate((arr, arr_current), axis = 0)
        large_string = ' '.join(arr)
        list_tocken = self.simple_preproc(large_string)
        values, counts = np.unique(list_tocken, return_counts = True)
        vocab = dict(zip(values, counts))
        tokens = self._filter_rare_words(vocab, min_occurancies).keys()
        return list(tokens)

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        if pd.__version__ > '1.3.0':
          embedings_arr = pd.read_csv(file_path, sep=' ', on_bad_lines='skip', dtype=object,  encoding='utf-8', quoting=3, header = None).to_numpy()
        else:
          embedings_arr = pd.read_csv(file_path, sep=' ', error_bad_lines=False, dtype=object,  encoding='utf-8', quoting=3, header = None).to_numpy()
        embedings_dict = {embedings_arr[i,0]: embedings_arr[i,1:].tolist() for i in range(embedings_arr.shape[0])}
        return embedings_dict

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        np.random.seed(random_seed)
        list_of_tockens = ['PAD', 'OOV'] + inner_keys # PAD [index] = 0, OOV [index] = 1
        embedings_dict = self._read_glove_embeddings(file_path)
        N = len(list_of_tockens)
        D = len(embedings_dict[next(iter(embedings_dict))])
        # PAD OOV exceptions
        embedings_dict['PAD'] = np.zeros(D).tolist()
        embedings_dict['OOV'] = np.random.uniform(low = -rand_uni_bound, high = rand_uni_bound,size = D).tolist()
        matrix = np.empty((N,D))

        vocab = {}
        unk_words =[]
        for index, tocken in enumerate(list_of_tockens):
          vocab[tocken] = index
          if tocken in embedings_dict.keys():
            matrix[index] = np.array(embedings_dict[tocken])
          else:
            matrix[index] = np.random.uniform(low = -rand_uni_bound, high = rand_uni_bound,size = D)
            unk_words.append(tocken)
        return (matrix, vocab, unk_words)

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words
        
    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int, num_of_triplets: int = 8_000
                                    ) -> List[List[Union[str, float]]]:
      inp_df_select = inp_df[['id_left', 'id_right', 'label']]
      np.random.seed(seed)
      neg_frac = 0.3

      df_train_same = (inp_df_select
                    .query("label == 1")
                    .sample(frac=1)
                    .groupby('id_left').first().reset_index()
                    )

      df_train_similar = (inp_df_select
                    .query("label == 0")
                    .sample(frac=1)
                    .groupby('id_left').first().reset_index()
                    )

      df_train_less = (inp_df_select
                    .query("label == 0")
                    .groupby('id_left').first().reset_index()
                    .rename(columns = {'id_right' : 'id_third','label' : 'label_third'})
                    )

      df_label_1_0 = df_train_same.merge(df_train_less, on = 'id_left', how = 'left').query('~label_third.isna()').assign(target = 1)
      df_label_0_0 = df_train_similar.merge(df_train_less, on = 'id_left', how = 'left').query('~label_third.isna() & (id_third != id_right)').assign(target = 0.5)

      df_random = (inp_df_select.groupby('id_left').first().reset_index().sample(frac=neg_frac))
      left_anti = df_random[['id_left', 'id_right']].values.reshape(-1)

      random_index_col = np.random.choice(['id_left', 'id_right'])
      df_random_neg = (inp_df_select
                        .query("~id_left.isin(@left_anti) & ~id_right.isin(@left_anti)")
                        .sample(n = df_random.shape[0])
                        [[random_index_col]]
                        .assign(label_third = -1)).to_numpy()
      df_random[['id_third', 'label_third']] = df_random_neg
      df_label_pos_neg = df_random.assign(target = 1)

      df_label_1_0 = df_label_1_0.sample(n = int(num_of_triplets / 3))
      df_label_0_0 = df_label_0_0.sample(n = int(num_of_triplets / 3))

      left_anti = pd.concat([df_label_1_0,df_label_0_0])[['id_left', 'id_right']].values.reshape(-1)
      df_label_pos_neg = (df_label_pos_neg.query("~id_left.isin(@left_anti) & ~id_right.isin(@left_anti)")
                                        .sample(n = int(num_of_triplets / 3)))
      result = pd.concat([df_label_1_0, df_label_0_0, df_label_pos_neg])[['id_left', 'id_right', 'id_third', 'target']]
      return result.to_numpy().tolist()
        
    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                          min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
      columns = ['id_left', 'id_right', 'label']
      inp_df_select = inp_df[columns]
    
      inf_df_group_sizes = inp_df_select.groupby('id_left').size()
      glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
    
      inp_df_rel = inp_df_select.query('label == 1').query("id_left in @glue_dev_leftids_to_use").assign(label = 2).to_numpy()
      inp_df_same = inp_df_select.query('label == 0').query("id_left in @glue_dev_leftids_to_use").assign(label = 1).to_numpy()
    
      values, freq = np.unique(inp_df[inp_df['id_left'].isin(glue_dev_leftids_to_use)][['id_left']].values.ravel('K'), return_counts=True)
      freq = np.minimum(freq, fill_top_to)
      n_elements, n_rows = values.shape[0], fill_top_to
    
      rnd_choise_df_list = np.empty((n_elements, n_rows))
    
      for index in range(values.shape[0]):
        rnd_choise_df = inp_df.query(f"(id_left != {values[index]}) & (id_right != {values[index]})")['id_left'].values
        rnd_choise_df = np.random.permutation(rnd_choise_df)[0: fill_top_to]
        rnd_choise_df_list[index] = rnd_choise_df
    
      vec_to_matrix = np.tile(values, fill_top_to).reshape(-1,values.shape[0]).T
      zero_label_matrix = np.zeros((values.shape[0], fill_top_to))
      rnd_choise = np.stack((vec_to_matrix,rnd_choise_df_list, zero_label_matrix), axis = -1)
      rnd_choise = rnd_choise.reshape(rnd_choise.shape[0]*rnd_choise.shape[1], rnd_choise.shape[2])
    
      result = pd.DataFrame(np.vstack((rnd_choise, inp_df_rel, inp_df_same)), columns = columns)
    
      result['rank'] = result.sort_values(['id_left', 'label'], ascending = False).groupby('id_left').cumcount() + 1
      result = result.query("rank <= 15").drop('rank', axis = 1).sort_values(['id_left', 'label'], ascending = [True, False])
      result[['id_left', 'id_right']] = result[['id_left', 'id_right']].astype(int).astype(str)
      return result.to_numpy().tolist()

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        def dcg(ys_true, ys_pred):
          ys_true = torch.Tensor(ys_true)
          ys_pred = torch.Tensor(ys_pred)
          _, argsort = torch.sort(torch.Tensor(ys_pred), descending=True, dim=0)
          argsort = argsort[:ndcg_top_k]
          ys_true_sorted = ys_true[argsort]
          ret = 0
          for i, l in enumerate(ys_true_sorted, 1):
              ret += (2 ** l - 1) / math.log2(1 + i)
          return ret
        ideal_dcg = dcg(ys_true, ys_true)
        pred_dcg = dcg(ys_true, ys_pred)

        return (pred_dcg / ideal_dcg).item()

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        ndsgs = []
        for epoch in range(n_epochs):
          if epoch % self.change_train_loader_ep == 0:
            train = self.sample_data_for_train_iter(self.glue_train_df, epoch)
            train_triplets = TrainTripletsDataset(train,
                                                  self.idx_to_text_mapping_train,
                                                  self.vocab,
                                                  self.vocab['OOV'],
                                                  self.simple_preproc)

            train_dataloader = torch.utils.data.DataLoader(train_triplets,
                                                           batch_size=self.dataloader_bs,
                                                           num_workers=0,
                                                           collate_fn=collate_fn, shuffle=False)
            for batch in train_dataloader:
              inp_1, inp_2, target = batch
              preds = self.model(inp_1, inp_2)
              loss = criterion(preds, target)
              loss.backward()
              opt.step()
            ndsg = self.valid(self.model, self.val_dataloader)
            ndsgs.append(ndsg)
            print(ndsgs)
            if ndsg > 0.925:
                print('break')
                break
    pass