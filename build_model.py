import builder
import pickle, torch, json

glue_qqp_dir = './data/QQP'
glove_path = './data/glove.6B.50d.txt'

pickle_file = './knrm_mlp.bin'
state_dict = './state_dict'
VOCAB_PATH = './data.json'

# build KNRM model
builder = builder.Solution(glue_qqp_dir,glove_path)
builder.train(100)

# save model, save embeddings
model = builder.model
with open(pickle_file, 'wb') as file:
    pickle.dump(model, file)
torch.save(builder.model.embeddings.state_dict(), state_dict)

# save tocken vocab
data = builder.vocab
with open(VOCAB_PATH, 'w') as json_file:
    json.dump(data, json_file, indent=4)