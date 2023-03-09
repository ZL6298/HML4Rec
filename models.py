import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from torch.autograd import Variable

class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_actor = config['num_actor']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim, 
            max_norm = 1.
        )
        
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim, 
            max_norm = 1.
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim,
            max_norm = 1.
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim,
            max_norm = 1.
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim,
            max_norm = 1.
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
    
class Rec_Model(torch.nn.Module):
    def __init__(self, config):
        super(Rec_Model, self).__init__()

        self.store_parameters()        
        self.store_top_parameters()
        
        self.user_emb = user(config)
        self.item_emb = item(config)
        
        self.fc_r1 = nn.Linear(config["embedding_dim"]*8, config["embedding_dim"]*4)
        self.fc_r2 = nn.Linear(config["embedding_dim"]*4, config["embedding_dim"]*2)
        
        self.fc_m1 = nn.Linear(config["embedding_dim"]*2, config["embedding_dim"])
        self.fc_m2 = nn.Linear(config["embedding_dim"], 4)
        
        
        self.fc_out = nn.Linear(4, 1)
        
        self.fc_Res = nn.Linear(config["embedding_dim"]*2, 4)
        
        self.local_update_target_weight_name = ['fc_r1.weight', 'fc_r1.bias', 'fc_r2.weight', 'fc_r2.bias']
        self.market_update_target_weight_name = ['fc_m1.weight', 'fc_m1.bias', 'fc_m2.weight', 'fc_m2.bias']
        
        
    def forward(self, x):
                   
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_vec = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_vec = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        concat_layer = torch.cat((user_vec, item_vec), 1)
        h_r1 = F.relu(self.fc_r1(concat_layer))
        h_r2 = F.relu(self.fc_r2(h_r1))
        
        h_m1 = F.relu(self.fc_m1(h_r2))
        h_m2 = F.relu(self.fc_m2(h_m1))
        
        out = self.fc_out(h_m2 + F.relu(self.fc_Res(h_r2)))
        
        sim = F.sigmoid(out)  
        
        return sim
    
    def store_parameters(self):
        self.keep_weight = deepcopy(self.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()
    
    def store_top_parameters(self):
        self.keep_weights_top = deepcopy(self.state_dict())
            