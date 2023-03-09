import numpy as np
import pandas as pd 
import datetime 
import random
import torch
import math
import sys
import time
import os
from copy import deepcopy
import warnings
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from Data_loader import *
from MeLU import *

warnings.filterwarnings('ignore')
random.seed(50)
np.random.seed(6298)
default_info = {
    'movielens': {'n_y': 5, 'u_in_dim': 3, 'i_in_dim': 4},
    'bookcrossing': {'n_y': 10, 'u_in_dim': 2, 'i_in_dim': 3}
}

def prepare_dataset_MAMO():
    train_order, valid_order, test_order, movie_dict, user_dict, Tr_rated_dict, Rated_dict, condidate_item = prepare_dataset()
    Tr_user = Build_user_list(train_order)
    V_user = Build_user_list(valid_order)
    Tr_rated_dict = {}
    for users in Tr_user:
        bought_list = list(train_order[train_order.user_id == users].movie_id.values)
        Tr_rated_dict[users] = bought_list
    
    V_rated_dict = {}
    for users in V_user:
        bought_list = list(valid_order[valid_order.user_id == users].movie_id.values)
        V_rated_dict[users] = bought_list
    index = 0    
    dict_user_id2index_Tr = {}
    for user_id in Tr_user:
        dict_user_id2index_Tr[user_id] = index
        index += 1
    index = 0
    dict_user_id2index_V = {}
    for user_id in V_user:
        dict_user_id2index_V[user_id] = index
        index += 1
    support_set_x, support_set_y, query_set_x, query_set_y = generate_dataset(Tr_user, 0, Tr_rated_dict, 5)
    valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y = generate_dataset(V_user, 1, V_rated_dict, 5, Tr_rated_dict)
    return train_order, valid_order, test_order, movie_dict, user_dict, Tr_rated_dict, \
           Rated_dict, dict_user_id2index_Tr, dict_user_id2index_V, condidate_item, \
           support_set_x, support_set_y, query_set_x, query_set_y,\
           valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y,\
           Tr_user, V_user

class RecMAM(torch.nn.Module):
    def __init__(self, embedding_dim, n_y, n_layer, activation='sigmoid', classification=True):
        super(RecMAM, self).__init__()
        self.input_size = embedding_dim * 2

        self.mem_layer = torch.nn.Linear(self.input_size, self.input_size)

        fcs = []
        last_size = self.input_size

        for i in range(n_layer - 1):
            out_dim = int(last_size / 4)
            linear_model = torch.nn.Linear(last_size, out_dim)
            fcs.append(linear_model)
            last_size = out_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, 2), activation_func("softmax")]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        out0 = self.mem_layer(x)
        out = self.fc(out0)
        out = self.final_layer(out)
        return out[:,0]
    
class MLItemLoading(torch.nn.Module):
    def __init__(self, config):
        super(MLItemLoading, self).__init__()
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

    def forward(self, x, vars=None):
        
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actors_idx = Variable(x[:, 2212:10242], requires_grad=False)
        
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class MLUserLoading(torch.nn.Module):
    def __init__(self, config):
        super(MLUserLoading, self).__init__()
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

    def forward(self, x):
        #print(x)
        gender_idx = Variable(x[:, 0], requires_grad=False)
        age_idx = Variable(x[:, 1], requires_grad=False)
        occupation_idx = Variable(x[:, 2], requires_grad=False)
        area_idx = Variable(x[:, 3], requires_grad=False)
        
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
    
class ItemEmbedding(torch.nn.Module):
    def __init__(self, n_layer, in_dim, embedding_dim, activation='sigmoid'):
        super(ItemEmbedding, self).__init__()
        self.input_size = in_dim

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size / 2)

        for i in range(n_layer - 1):
            linear_model = torch.nn.Linear(last_size, hid_dim)
            linear_model.bias.data.fill_(0.0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out

# user embedding
class UserEmbedding(torch.nn.Module):
    def __init__(self, n_layer, in_dim, embedding_dim, activation='sigmoid'):
        super(UserEmbedding, self).__init__()
        self.input_size = in_dim

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size / 2)

        for i in range(n_layer - 1):
            linear_model = torch.nn.Linear(last_size, hid_dim)
            linear_model.bias.data.fill_(0.0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out
    
class LOCALUpdate:
    def __init__(self, your_model, u_idx, dataset, sup_size, que_size, 
                 bt_size, n_loop, update_lr, if_test, top_k, device,
                 movie_dict, user_dict,
                 dict_user_id2index_Tr, dict_user_id2index_V,
                 support_set_x, support_set_y, query_set_x, query_set_y, 
                 valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y,
                 dict_user_id2index_T = None, sup_s_x = None, sup_s_y = None, qry_x = None
                ):

        self.support_set_x = support_set_x 
        self.support_set_y = support_set_y 
        self.query_set_x = query_set_x 
        self.query_set_y = query_set_y 
        self.valid_s_set_x = valid_s_set_x 
        self.valid_s_set_y = valid_s_set_y 
        self.valid_q_set_x = valid_q_set_x 
        self.valid_q_set_y = valid_q_set_y 
        self.dict_user_id2index_Tr = dict_user_id2index_Tr 
        self.dict_user_id2index_V = dict_user_id2index_V
        self.dict_user_id2index_T = dict_user_id2index_T 
        self.sup_s_x = sup_s_x 
        self.sup_s_y = sup_s_y 
        self.qry_x = qry_x
        self.movie_dict, self.user_dict = movie_dict, user_dict
        
        
        if if_test != 2:
            self.s_x, self.s_y, self.q_x, self.q_y = self.get_training_data_per_user(u_idx, if_test)
        if if_test == 2:
            self.s_x, self.s_y, self.q_x = self.get_rec_data_per_user(u_idx, if_test)
        user_data = UserDataLoader(self.s_x, self.s_y)
        self.user_data_loader = DataLoader(user_data, batch_size=len(self.s_x))
        
        self.model = your_model

        self.update_lr = update_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.update_lr)

        self.loss_fn = torch.nn.BCELoss()

        self.n_loop = n_loop
        self.top_k = top_k

        self.device = device

    def train(self):
        for i in range(self.n_loop):
            for i_batch, (x1, x2, y) in enumerate(self.user_data_loader):
                x1_feature = self.map_user_feature(x1)
                x2_feature = self.map_item_feature(x2)
                pred_y = self.model(x1_feature, x2_feature)
                
                loss = self.loss_fn(pred_y, y[0].float().unsqueeze(0))
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
        
        q_x1 = [x1[0] for x1 in self.q_x]
        q_x2 = [x2[1] for x2 in self.q_x]
        q_x1_feature = self.map_user_feature(q_x1)
        q_x2_feature = self.map_item_feature(q_x2)
        q_pred_y = self.model(q_x1_feature, q_x2_feature)
        self.optimizer.zero_grad()
                
        loss = self.loss_fn(q_pred_y, torch.Tensor(self.q_y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        u_grad, i_grad, r_grad = self.model.get_grad()
        return u_grad, i_grad, r_grad, loss.data

    def test(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (x1, x2, y) in enumerate(self.user_data_loader):
                x1_feature = self.map_user_feature(x1)
                x2_feature = self.map_item_feature(x2)
                pred_y = self.model(x1_feature, x2_feature)
                loss = self.loss_fn(pred_y, y[0].float().unsqueeze(0))
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        # D.I.Y your calculation for the results
        q_x1 = [x1[0] for x1 in self.q_x]
        q_x2 = [x2[1] for x2 in self.q_x]
        q_x1_feature = self.map_user_feature(q_x1)
        q_x2_feature = self.map_item_feature(q_x2)        
        q_pred_y = self.model(q_x1_feature, q_x2_feature)  # on query set
        loss = self.loss_fn(q_pred_y, torch.Tensor(self.q_y))
        return loss.data, q_pred_y.data
    
    def Rec(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (x1, x2, y) in enumerate(self.user_data_loader):
                x1_feature = self.map_user_feature(x1)
                x2_feature = self.map_item_feature(x2)
                pred_y = self.model(x1_feature, x2_feature)
                loss = self.loss_fn(pred_y, y[0].float().unsqueeze(0))
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        # D.I.Y your calculation for the results
        q_x1 = [x1[0] for x1 in self.q_x]
        q_x2 = [x2[1] for x2 in self.q_x]
        q_x1_feature = self.map_user_feature(q_x1)
        q_x2_feature = self.map_item_feature(q_x2)
        q_pred_y = self.model(q_x1_feature, q_x2_feature)  # on query set      
        return q_pred_y.data
    
    def get_training_data_per_user(self, u_id, if_test):
        if if_test == 0:
            user_index = self.dict_user_id2index_Tr[u_id]
            support_x, support_y = self.support_set_x[user_index], self.support_set_y[user_index]
            query_x, query_y = self.query_set_x[user_index], self.query_set_y[user_index]
        if if_test == 1:
            user_index = self.dict_user_id2index_V[u_id]
            support_x, support_y = self.valid_s_set_x[user_index], self.valid_s_set_y[user_index]
            query_x, query_y = self.valid_q_set_x[user_index], self.valid_q_set_y[user_index]        
        return support_x, support_y, query_x, query_y

    def get_rec_data_per_user(self, u_id, if_test):
        if if_test == 2:
            user_index = self.dict_user_id2index_T[u_id]
            support_x, support_y = self.sup_s_x[user_index], self.sup_s_y[user_index]
            query_x = self.qry_x[user_index]       
        return support_x, support_y, query_x
    
    def map_user_feature(self, users):
        x_features = None
        for u_id in users:
            try:
                x_features = torch.cat((x_features, self.user_dict[int(u_id)]),0)
            except:
                x_features = self.user_dict[int(u_id)]

        return x_features

    def map_item_feature(self, items):
        x_features = torch.LongTensor(len(items), 10242)
        i = 0
        for i_id in items:
            x_features[i] = self.movie_dict[int(i_id)] 
            i += 1
        return x_features

class UserDataLoader(Dataset):
    def __init__(self, x, y, transform=None):
        self.x1 = [x1[0] for x1 in x]
        self.x2 = [x2[1] for x2 in x]
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        user_info = self.x1[idx]
        item_info = self.x2[idx]
        ratings = self.y[idx]

        return user_info, item_info, ratings

class MAMRec:
    def __init__(self, config_settings, Tr_user, V_user, movie_dict, user_dict, dataset='movielens'):

        self.dataset = dataset
        self.support_size = config_settings['support_size']
        self.query_size = config_settings['query_size']
        self.n_epoch = config_settings['n_epoch']
        self.n_inner_loop = config_settings['n_inner_loop']
        self.batch_size = config_settings['batch_size']
        self.n_layer = config_settings['n_layer']
        self.embedding_dim = config_settings['embedding_dim']
        self.rho = config_settings['rho']  # local learning rate
        #self.rho = 0.00001
        self.lamda = config_settings['lamda']  # global learning rate
        #self.lamda = 0.0001
        self.tao = config_settings['tao']  # hyper-parameter for initializing personalized u weights
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config_settings['cuda_option'] if self.USE_CUDA else "cpu")
        self.n_k = config_settings['n_k']
        self.alpha = config_settings['alpha']
        self.beta = config_settings['beta']
        self.gamma = config_settings['gamma']
        self.active_func = config_settings['active_func']
        self.rand = config_settings['rand']
        self.random_state = config_settings['random_state']
        self.split_ratio = config_settings['split_ratio']

        # load dataset
        self.train_users, self.test_users = Tr_user, V_user
        self.movie_dict, self.user_dict = movie_dict, user_dict
        
        self.x1_loading, self.x2_loading =  MLUserLoading(config_settings), \
                                            MLItemLoading(config_settings)
        
        self.n_y = default_info[dataset]['n_y']

        # Embedding model
        self.UEmb = UserEmbedding(self.n_layer, 4 * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)
        self.IEmb = ItemEmbedding(self.n_layer, default_info[dataset]['i_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)

        # rec model
        self.rec_model = RecMAM(self.embedding_dim, self.n_y, self.n_layer, 
                                activation=self.active_func, classification=False).to(self.device)

        # whole model
        self.model = BASEModel(self.x1_loading, self.x2_loading, self.UEmb, self.IEmb, self.rec_model).to(self.device)

        self.phi_u, self.phi_i, self.phi_r = self.model.get_weights()

        self.FeatureMEM = FeatureMem(self.n_k, 4 * self.embedding_dim,
                                     self.model, device=self.device)
        self.TaskMEM = TaskMem(self.n_k, self.embedding_dim, device=self.device)

        self.train = self.train_with_meta_optimization
        self.test = self.test_with_meta_optimization

        #self.train()

    def train_with_meta_optimization(self, dict_user_id2index_Tr, 
                                     dict_user_id2index_V, support_set_x, support_set_y, query_set_x, query_set_y,
                                     valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y):
        for i in range(self.n_epoch):
            #u_grad_sum, i_grad_sum, r_grad_sum = self.model.get_zero_weights()

            # On training dataset
            for mini_batch in range(len(self.train_users) // self.batch_size):
                loss_batch = [] 
                starttime = time.time()
                u_grad_sum, i_grad_sum, r_grad_sum = self.model.get_zero_weights()
                for u in self.train_users[mini_batch * self.batch_size: (mini_batch+1) * self.batch_size]: 
                    # init local parameters: theta_u, theta_i, theta_r at (self.model.init_u_mem_weights)
                    bias_term, att_values = self.user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading,
                                                          self.alpha)                                      
                    self.model.init_u_mem_weights(self.phi_u, bias_term, self.tao, self.phi_i, self.phi_r)
                    self.model.init_ui_mem_weights(att_values, self.TaskMEM)
                    user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                              self.n_inner_loop, self.rho, 0, 3, self.device, 
                                              self.movie_dict, self.user_dict,
                                              dict_user_id2index_Tr, dict_user_id2index_V, support_set_x, 
                                              support_set_y, query_set_x, query_set_y,
                                              valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y)
                    u_grad, i_grad, r_grad, loss = user_module.train()

                    u_grad_sum, i_grad_sum, r_grad_sum = grads_sum(u_grad_sum, u_grad), grads_sum(i_grad_sum, i_grad), \
                                                         grads_sum(r_grad_sum, r_grad)

                    self.FeatureMEM.write_head(u_grad, self.beta)
                    u_mui = self.model.get_ui_mem_weights()
                    self.TaskMEM.write_head(u_mui[0], self.gamma)
                    
                    #print("single_user_loss:", loss)
                    
                    loss_batch.append(loss)
                loss_batch = torch.stack(loss_batch).mean()
                

                self.phi_u, self.phi_i, self.phi_r = maml_train(self.phi_u, self.phi_i, self.phi_r,
                                                                u_grad_sum, i_grad_sum, r_grad_sum, self.lamda)

                
                
                
                
                endtime = time.time()
                sys.stdout.write("\r epoch:{0}  {1}/{2}  batch_loss_query_set: {3} time: {4}"
                                     .format(i, mini_batch, len(self.train_users) // self.batch_size, loss_batch, endtime - starttime))
                sys.stdout.flush() 

            valid_loss,_ = self.test_with_meta_optimization(dict_user_id2index_Tr, 
                                    dict_user_id2index_V, support_set_x, support_set_y, query_set_x, query_set_y,
                                    valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y)
            print()
            print('epoch_' + str(i) + ":  epoch loss_query_set = " + str(valid_loss))
        torch.save(model, "MAMO")

    def test_with_meta_optimization(self, dict_user_id2index_Tr, 
                                    dict_user_id2index_V, support_set_x, support_set_y, query_set_x, query_set_y,
                                    valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y
                                   ):
        best_phi_u, best_phi_i, best_phi_r = self.model.get_weights()
        loss_batch = [] 
        pred_list = []
        for u in self.test_users:
            
            bias_term, att_values = self.user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading,
                                                  self.alpha)
            self.model.init_u_mem_weights(best_phi_u, bias_term, self.tao, best_phi_i, best_phi_r)
            self.model.init_ui_mem_weights(att_values, self.TaskMEM)

            self.model.init_weights(best_phi_u, best_phi_i, best_phi_r)
            
            user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                      self.n_inner_loop, self.rho, 1, 3, self.device, 
                                      self.movie_dict, self.user_dict,
                                      dict_user_id2index_Tr, dict_user_id2index_V, support_set_x, 
                                      support_set_y, query_set_x, query_set_y,
                                      valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y)
            
            loss, pred_rslt = user_module.test()
            loss_batch.append(loss)
            pred_list.append(pred_rslt)
        loss_batch = torch.stack(loss_batch).mean()
        pred_list = torch.stack(pred_list)
        return loss_batch, pred_list
    
    def Rec_with_meta_optimization(self, Rec_users, dict_user_id2index_Tr, dict_user_id2index_V,
                                   support_set_x, support_set_y, query_set_x, query_set_y, 
                                   valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y,
                                   dict_user_id2index_T, q_sub_s_x, q_sub_s_y, q_x
                                  ):
        best_phi_u, best_phi_i, best_phi_r = self.model.get_weights()
        pred_reclist = {}
        for u in Rec_users:
            
            bias_term, att_values = self.user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading,
                                                  self.alpha)
            self.model.init_u_mem_weights(best_phi_u, bias_term, self.tao, best_phi_i, best_phi_r)
            self.model.init_ui_mem_weights(att_values, self.TaskMEM)

            self.model.init_weights(best_phi_u, best_phi_i, best_phi_r)
            
            user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                      self.n_inner_loop, self.rho, 2, 3, self.device,
                                      self.movie_dict, self.user_dict,
                                      dict_user_id2index_Tr, dict_user_id2index_V,
                                      support_set_x, support_set_y, query_set_x, query_set_y, 
                                      valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y,
                                      dict_user_id2index_T, q_sub_s_x, q_sub_s_y, q_x
                                      )
            condidate_item_user = [x[1] for x in q_x[dict_user_id2index_T[u]]]
            pred_rslt = user_module.Rec()
            pred_score = dict(zip(condidate_item_user, pred_rslt.data))
            pred_score = sorted(pred_score.items(),key = lambda x:x[1],reverse = True)
            pred_score_top100 = pred_score[:100]
            for k in range(100):
                pred_score_top100[k] = pred_score_top100[k][0]  
            pred_reclist[u] = pred_score_top100
        return pred_reclist
    
    def user_mem_init(self, u_id, dataset, device, feature_mem, loading_model, alpha):
        u_x1 = torch.LongTensor(self.user_dict[u_id])
        pu = loading_model(u_x1)
        personalized_bias_term, att_values = feature_mem.read_head(pu, alpha)
        del u_x1, pu
        return personalized_bias_term, att_values
    
class FeatureMem:
    def __init__(self, n_k, u_emb_dim, base_model, device):
        self.n_k = n_k
        self.base_model = base_model
        self.p_memory = torch.randn(n_k, u_emb_dim, device=device).normal_()  # on device
        u_param, _, _ = base_model.get_weights()
        self.u_memory = []
        for i in range(n_k):
            bias_list = []
            for param in u_param:
                bias_list.append(param.normal_(std=0.05))
            self.u_memory.append(bias_list)
        self.att_values = torch.zeros(n_k).to(device)
        self.device = device

    def read_head(self, p_u, alpha, train=True):
        # get personalized mu
        att_model = Attention(self.n_k).to(self.device)
        attention_values = att_model(p_u, self.p_memory).to(self.device)  # pu on device
        personalized_mu = get_mu(attention_values, self.u_memory, self.base_model, self.device)
        # update mp
        transposed_att = attention_values.reshape(self.n_k, 1)
        product = torch.mm(transposed_att, p_u)
        if train:
            self.p_memory = alpha * product + (1-alpha) * self.p_memory
        self.att_values = attention_values
        return personalized_mu, attention_values

    def write_head(self, u_grads, lr):
        update_mu(self.att_values, self.u_memory, u_grads, lr)


class TaskMem:
    def __init__(self, n_k, emb_dim, device):
        self.n_k = n_k
        self.memory_UI = torch.rand(n_k, emb_dim *2, emb_dim*2, device=device).normal_()
        self.att_values = torch.zeros(n_k)

    def read_head(self, att_values):
        self.att_values = att_values
        return get_mui(att_values, self.memory_UI, self.n_k)

    def write_head(self, u_mui, lr):
        update_values = update_mui(self.att_values, self.n_k, u_mui)
        self.memory_UI = lr* update_values + (1-lr) * self.memory_UI


def cosine_similarity(input1, input2):
    query_norm = torch.sqrt(torch.sum(input1**2+0.00001, 1))
    doc_norm = torch.sqrt(torch.sum(input2**2+0.00001, 1))

    prod = torch.sum(torch.mul(input1, input2), 1)
    norm_prod = torch.mul(query_norm, doc_norm)

    cos_sim_raw = torch.div(prod, norm_prod)
    return cos_sim_raw


class Attention(torch.nn.Module):
    def __init__(self, n_k, activation='relu'):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k, activation_func(activation))
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        inputs = cosine_similarity(expanded_pu, mp)
        fc_layers = self.fc_layer(inputs)
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values


def get_mu(att_values, mu, model, device):
    mu0,_,_ = model.get_zero_weights()
    attention_values = att_values.reshape(len(mu),1)
    for i in range(len(mu)):
        for j in range(len(mu[i])):
            mu0[j] += attention_values[i] * mu[i][j].to(device)
    return mu0


def update_mu(att_values, mu, grads, lr):
    att_values = att_values.reshape(len(mu), 1)
    for i in range(len(mu)):
        for j in range(len(mu[i])):
            mu[i][j] = lr * att_values[i] * grads[j] + (1-lr) * mu[i][j]


def get_mui(att_values, mui, n_k):
    attention_values = att_values.reshape(n_k, 1, 1)
    attend_mui = torch.mul(attention_values, mui)
    u_mui = attend_mui.sum(dim=0)
    return u_mui


def update_mui(att_values, n_k, u_mui):
    repeat_u_mui = u_mui.unsqueeze(0).repeat(n_k, 1, 1)
    attention_tensor = att_values.reshape(n_k, 1, 1)
    attend_u_mui = torch.mul(attention_tensor, repeat_u_mui)
    return attend_u_mui

def to_torch(in_list):
    return torch.from_numpy(np.array(in_list))

# =============================================
def get_params(param_list):
    params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.data)
            params.append(value)
            del value
        count += 1
    return params


def get_zeros_like_params(param_list):
    zeros_like_params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(torch.zeros_like(param.data))
            zeros_like_params.append(value)
        count += 1
    return zeros_like_params


def init_params(param_list, init_values):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count])
            init_count += 1
        count += 1


def init_u_mem_params(param_list, init_values, bias_term, tao):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count]-tao*bias_term[init_count])
            init_count += 1
        count += 1


def init_ui_mem_params(param_list, init_values):
    count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values)
        count += 1


def get_grad(param_list):
    count = 0
    param_grads = []
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.grad)
            param_grads.append(value)
            del value
        count += 1
    return param_grads


def grads_sum(raw_grads_list, new_grads_list):
    return [raw_grads_list[i]+new_grads_list[i] for i in range(len(raw_grads_list))]


def update_parameters(params, grads, lr):
    return [params[i] - lr*grads[i] for i in range(len(params))]


# ===============================================
def activation_func(name):
    name = name.lower()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == 'softmax':
        return torch.nn.Softmax()
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(0.1)
    else:
        return torch.nn.Sequential()
    
class BASEModel(torch.nn.Module):
    def __init__(self, input1_module, input2_module, embedding1_module, embedding2_module, rec_module):
        super(BASEModel, self).__init__()

        self.input_user_loading = input1_module
        self.input_item_loading = input2_module
        self.user_embedding = embedding1_module
        self.item_embedding = embedding2_module
        self.rec_model = rec_module

    def forward(self, x1, x2):
        pu, pi = self.input_user_loading(x1), self.input_item_loading(x2)
        eu, ei = self.user_embedding(pu), self.item_embedding(pi)
        rec_value = self.rec_model(eu, ei)
        return rec_value

    def get_weights(self):
        u_emb_params = get_params(self.user_embedding.parameters())
        i_emb_params = get_params(self.item_embedding.parameters())
        rec_params = get_params(self.rec_model.parameters())
        return u_emb_params, i_emb_params, rec_params

    def get_zero_weights(self):
        zeros_like_u_emb_params = get_zeros_like_params(self.user_embedding.parameters())
        zeros_like_i_emb_params = get_zeros_like_params(self.item_embedding.parameters())
        zeros_like_rec_params = get_zeros_like_params(self.rec_model.parameters())
        return zeros_like_u_emb_params, zeros_like_i_emb_params, zeros_like_rec_params

    def init_weights(self, u_emb_para, i_emb_para, rec_para):
        init_params(self.user_embedding.parameters(), u_emb_para)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def get_grad(self):
        u_grad = get_grad(self.user_embedding.parameters())
        i_grad = get_grad(self.item_embedding.parameters())
        r_grad = get_grad(self.rec_model.parameters())
        return u_grad, i_grad, r_grad

    def init_u_mem_weights(self, u_emb_para, mu, tao, i_emb_para, rec_para):
        init_u_mem_params(self.user_embedding.parameters(), u_emb_para, mu, tao)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def init_ui_mem_weights(self, att_values, task_mem):
        # init the weights only for the mem layer
        u_mui = task_mem.read_head(att_values)
        init_ui_mem_params(self.rec_model.mem_layer.parameters(), u_mui)

    def get_ui_mem_weights(self):
        return get_params(self.rec_model.mem_layer.parameters())
    
def maml_train(raw_phi_u, raw_phi_i, raw_phi_r, u_grad_list, i_grad_list, r_grad_list, global_lr):
    phi_u = update_parameters(raw_phi_u, u_grad_list, global_lr)
    phi_i = update_parameters(raw_phi_i, i_grad_list, global_lr)
    phi_r = update_parameters(raw_phi_r, r_grad_list, global_lr)
    return phi_u, phi_i, phi_r

def MAMO_Recommending(test_order, Tr_rated_dict, Rated_dict, condidate_item, dict_user_id2index_Tr, dict_user_id2index_V,
                                   support_set_x, support_set_y, query_set_x, query_set_y, 
                                   valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y):
    bias_list = [0, 14, 28]
    w_u = [0, 1]
    for f_w_u in w_u:
        print("===================================================================================")
        for bias in bias_list:

            start_date = 0 + bias
            mid_date = 7 + bias
            end_date = 14 + bias

            test_initial_day = test_order.days.min()
            q_sub_s_x, q_sub_s_y, Rec_user, q_item_list, Dict_T_purchuse_list\
            = MeLU_generate_dataset_for_Rec(test_order, Tr_rated_dict, bias, configs_MeLU['durations'], 
                                            configs_MeLU['min_pchs'], configs_MeLU['len_query_i'], f_w_u)

            model = torch.load("MAMO")          
            
            q_x = []
            for i in range(len(Rec_user)):
                q_set_q = []
                condidate_item_user = np.setdiff1d(condidate_item, Rated_dict[Rec_user[i]])
                condidate_item_user = np.union1d(condidate_item_user, Dict_T_purchuse_list[Rec_user[i]])                              
                for j in range(len(condidate_item_user)):
                    q_set_q.append([Rec_user[i], condidate_item_user[j]]) 
                q_x.append(q_set_q)

            dict_user_id2index_T = {}
            index = 0
            for user_id in Rec_user:
                dict_user_id2index_T[user_id] = index
                if index % 10000 == 0:
                    print(str(index)+ "/" + str(len(Rec_user)))
                index += 1

            pred_reclist = model.Rec_with_meta_optimization(Rec_user, dict_user_id2index_Tr, dict_user_id2index_V,
                                   support_set_x, support_set_y, query_set_x, query_set_y, 
                                   valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y,
                                   dict_user_id2index_T, q_sub_s_x, q_sub_s_y, q_x
                                                           )
            np.save("./Rec_result/MAMO_Reclist_bias{0}_wu{1}.npy".format(bias, f_w_u), pred_reclist, allow_pickle=True)
            print("bias_" + str(bias) + " end")
            print("------------------------------------------------------")
        print("========================================================================================")
        print()
        print()