import requests, json, pandas as pd
from typing import Dict, List, Tuple, Union, Callable, Optional
import builder 
# URL вашего Flask приложения
base_url = 'http://127.0.0.1:5000'

class Solution:
  # import and tockenize dara + debug functions
  def __init__(
      self,
      glue_qqp_dir,
      ):
    self.glue_qqp_dir = glue_qqp_dir
    self.glue_df_train = self.get_glue_df('train')
    self.glue_df_test = self.get_glue_df('dev')
    self.json_query = self._generate_json_server_query()
    self.json_documents = self._generate_json_server_documents()

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

  def _generate_json_server_query(self) -> Dict[str, List[str]]:
    # generating query for debug
    return {'query' : self.glue_df_test['text_left'].sample(10).tolist()}

  def _generate_json_server_documents(self) -> Dict[str, Dict[str, List[str]]]:
    # generating documents for debug
    left_dict = self.glue_df_train[['id_left', 'text_left']].set_index('id_left').to_dict()['text_left']
    # right_dict = self.glue_df_train[['id_right', 'text_right']].set_index('id_right').to_dict()
    # documents = left_dict.update(right_dict)
    return {'documents' : left_dict}   
  
def print_into_terminal(array):
   for i in array:
      print(i)
   pass

def query():
    url = f'{base_url}/query'
    payload = sol._generate_json_server_query()
    print_into_terminal(payload['query'])

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.json()

def update_index():

   url = f'{base_url}/update_index'

   payload = sol._generate_json_server_documents()
   response = requests.post(url, data=json.dumps(payload), headers=headers)
   return response.json()

if __name__ == '__main__':
    sol = Solution(builder.glue_qqp_dir)
    headers = {'Content-Type': 'application/json'}
    # query_text = "Tell me about FAISS."
    update_response = update_index()
    query_response = query()
    print('Update Response:', update_response)
    print_into_terminal(query_response['suggestions'])