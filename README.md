# Ranking_Model_Flask_service
As a final project, a suggestion system for similar questions was implemented using data from the Quora website. The search is conducted exclusively based on the main title without any clarifying details. The system is presented as a microservice based on Flask. The high-level pipeline and criteria can be described as follows:

First, the query is filtered by language (using the LangDetect library) — all queries for which the detected language is not "en" are excluded. Then, candidate questions are searched using FAISS (based on vector similarity) — in this part, it is proposed to limit vectorization to only those words whose embeddings are present in the original GLOVE vectors. These candidates are re-ranked by the KNRM model, after which up to 10 candidates are returned as the response.

The server implements two endpoints: one for queries (to search for similar questions) and one for creating the FAISS index.

/query — accepts a POST request. It should return a JSON where status='FAISS is not initialized!' if the solution has not been loaded with questions for searching using the second method.

Request format for query: a JSON request with a single key 'queries', the value of which is a list of strings with questions (Dict[str, List[str]]).

Response format (in case the index is created) — a JSON with two fields. lang_check describes whether the query was recognized as English (List[bool], True/False values), suggestions — List[Optional[List[Tuple[str, str]]]].

In this list, for each query from the query, you need to specify a list (up to 10) of found similar questions, where each question is represented as a Tuple, in which the first value is the text id (see below), and the second is the raw text of the similar question. If the language check fails (not English), or if there is some processing error, leave None in the list instead of the response (for example, [[(..., ...), (..., ...), ...], None, ... ]).

/update_index — accepts a POST request, in which the JSON contains a field documents, Dict[str, str] — all documents, where the key is the text id, and the value is the text itself. 200 seconds are given for preprocessing and creating the index. It is assumed that the initialization occurs only once, so there is no need to worry about calling this method again. The returned JSON should have two keys: status (ok, if everything went smoothly) and index_size, the value of which is a single integer representing the number of documents in the index.

A demonstration of the service's operation is implemented in the notebook.(demonstration.ipynb)


