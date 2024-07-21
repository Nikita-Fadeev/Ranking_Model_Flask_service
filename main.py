from flask import Flask, request, jsonify
from typing import Dict, List, Tuple, Union, Callable, Optional
from langdetect import detect

import nltk, re, math,json
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F

import pickle, faiss
from builder import GaussianKernel, KNRM


drive_dir = ''

VOCAB_PATH = drive_dir + 'knrm/data.json'
MLP_PATH  = drive_dir + 'knrm/knrm_mlp.bin'
EMB_PATH_KNRM = drive_dir + 'knrm/state_dict'

class FaissIndexer:
  # build faiss indexer based on documents
  def __init__(self,
      documents: np.array,
  ):
    self.documents = documents
    self.nd = self.documents.shape[0]
    self.N, self.D = self.documents.shape[1], self.documents.shape[2]
    # self.doc = self.documents.reshape(self.nd, self.N * self.D)
    # self.index = faiss.IndexFlatL2(self.N * self.D)
    self.doc = np.mean(self.documents, axis = 1)
    self.index = faiss.IndexFlatL2(self.D)
    self.index.add(self.doc)

  def search(self, query: np.array, k: np.array = 10) -> np.array:
    nq = query.shape[0]
    # quer = query.reshape(nq, self.N * self.D)
    quer = np.mean(query, axis = 1)
    D, I = self.index.search(quer, k)
    return I

class Advice:
  # contains FAISS indexer for choosing candidates, pretrained ranking KNRM model,
  # .get() return top nearest documents (Quora questions) based on query
  def __init__(self,
      MLP_PATH,
      VOCAB_PATH,
      EMB_PATH_KNRM,
      top_k_faiss = 15,
      top_k_knrm = 10,
      max_sentense_len = 30
  ):
    with open(MLP_PATH, 'rb') as file:
      self.model = pickle.load(file)

    self.embeddings = torch.load(EMB_PATH_KNRM)['weight'].numpy()
    self.model.embeddings.load_state_dict(torch.load(EMB_PATH_KNRM))
    self.vocab_glove = json.load(open(VOCAB_PATH))
    self.top_k_faiss = top_k_faiss
    self.top_k_knrm = top_k_knrm
    self.indexer = None
    self.max_sentense_len = max_sentense_len

  def _tockenize(self, df: pd.Series) -> List[str]:
    # str -> lower -> reduce punctuation -> tockenize through nltk
    df = df.str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
    return df.apply(lambda string : nltk.word_tokenize(string))

  def _convert_tockenize_series_to_numpy(self, series: pd.Series, empty_string = 'PAD') -> np.array:
    # list([tocken tocken tocken tocken], ... ) -> np.array([tocken tocken tocken tocken], ... )
    # converting differnt size list of tockens to same size vector

    # empty object array
    lens = series.str.len()
    max_len = lens.max()
    arr = np.zeros((lens.shape[0],max_len),object)
    # mask that explain length of each list
    mask = np.arange(max_len) < np.array(lens)[:,None]
    arr[mask] = np.concatenate(series.tolist())
    arr[~mask] = empty_string
    return arr

  def _replace_with_dict(self, arr, dict_, reverse = False) -> np.array:
      # [id, id, id] -> [dict[id], dict[id], dict[id]]
      # Extract out keys and values
      k = np.array(list(dict_.keys()))
      v = np.array(list(dict_.values()))
      # Get argsort indices
      sidx = k.argsort()
      ks = k[sidx]
      vs = v[sidx]
      return vs[np.searchsorted(ks,arr)]

  def convert_tockens_to_ids(self, tockens_array, vocab, max_len = 30) -> np.array:
    # [tocken, tocken] -> [id_tocken, id_tocken]
    arr = self._convert_tockenize_series_to_numpy(tockens_array)[:, : max_len]
    result = self._replace_with_dict(arr, vocab)
    # increase columns to max_len
    if result.shape[1] < max_len:
      N, D = result.shape[0], result.shape[1]
      zeros_array = np.zeros((N, max_len - D), dtype = int)
      result = np.concatenate((result, zeros_array), axis = 1)
    return result

  def _detect_eng(self, arr) -> np.array:
    # return mask where true is eng query
    detect_np = np.vectorize(detect)
    eng_mask = detect_np(arr) == 'en'
    return eng_mask

  def _replace_ids_to_embeddings(self, arr:np.array, embeddings: np.array) -> np.array:
    # [[id_tocken, id_tocken],  [id_tocken, id_tocken]] -> [[[emb], [emb]],  [[emb], [emb]]]
    return embeddings[arr]

  def update_index(self, json_documents: Dict) -> None:
    # init FAISS indexer
    self.documents_indexes = np.array(list(json_documents['documents'].keys()))
    self.documents_text = np.array(list(json_documents['documents'].values()))
    documents_tockenize = self._tockenize(pd.Series(self.documents_text))

    self.document_ids = self.convert_tockens_to_ids(documents_tockenize, self.vocab_glove, self.max_sentense_len)
    self.document_embed = self._replace_ids_to_embeddings(self.document_ids, self.embeddings)
    self.indexer = FaissIndexer(self.document_embed)
    
    pass

  def get(self, query: np.array, top_k_faiss: int = 15, top_k_knrm: int = 10):# -> Dict[str: List[Optional[List[Tuple[str, str]]]]]:
    # find nearest top_k_faiss documents and rank by KNRM model
    # return True/False array where query is Eng, suggestion like [(document_id, document_text), (..., ... ), ...]
    if self.indexer == None:
      raise ValueError('update_index() first')
    
    lang_check = self._detect_eng(query)
    query_tockenize = self._tockenize(pd.Series(query))
    query_ids = self.convert_tockens_to_ids(query_tockenize, self.vocab_glove, self.max_sentense_len)
    query_embed = self._replace_ids_to_embeddings(query_ids, self.embeddings)

    nearest_documents_ids = self.indexer.search(query_embed, top_k_faiss)
    suggestions = []
    # get top_k_knrm suggestions for each query
    for i in range(lang_check.shape[0]):
      if lang_check[i]:
        query_input = torch.Tensor(query_ids[i] * np.ones_like(self.document_ids[nearest_documents_ids[i]]))
        documents_input = torch.Tensor(self.document_ids[nearest_documents_ids[i]])

        input = {'query': query_input, 'document': documents_input}
        preds = self.model.predict(input)

        rank_knrm = torch.argsort(preds.reshape(-1), descending=True).numpy()
        top = nearest_documents_ids[i][rank_knrm][:top_k_knrm]

        id = self.documents_indexes[top].tolist()
        text = self.documents_text[top].tolist()

        suggestion = list(zip(id, text))
      else:
        suggestion = [[None, None]] * top_k_knrm 
      suggestions.append(suggestion)

    response = {'lang_check': lang_check.tolist(), 'suggestions': suggestions}
    return response

adviser = Advice(MLP_PATH, VOCAB_PATH,EMB_PATH_KNRM)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "<h1>Hello, World!<h1>"

@app.route('/ping')
def init():
    if adviser is not None :
        req = {'status' : 'ok'}
    else:
        req = {'status' : 'wait'}
    return jsonify(req)

@app.route('/query', methods=['POST'])
def load_questions():
    # find and return top N nearest document based on Quora questions dataset
    global questions
    data = request.get_json()
    if adviser.indexer is None:
       req = {'status': 'FAISS is not initialized!'}
    else:
        questions = data.get('query', [])
        req = adviser.get(questions)
    return jsonify(req)
    
@app.route('/update_index', methods=['POST'])
def update_index():
    # init FAISS indexer
    data = request.get_json()
    if data is not None:
        try:
            adviser.update_index(data)
            req = {'status' : 'ok'}
        except:
            req = {'status' : 'error'}
            return jsonify(req)
    else:
       req = {'status' : 'error: attach file to update index'}
    return jsonify(req)


if __name__ == '__main__':
   app.run(host='0.0.0.0', debug = True)