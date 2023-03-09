import numpy as np 
import pandas as pd 
import datetime
import random 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from collections import OrderedDict
from copy import deepcopy
from torch.autograd import Variable
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from Data_loader import *
from configs import *
from models import *

def Build_user_list(dataset):
    dataset = dataset.drop_duplicates(subset=["user_id","movie_id"],keep="first",inplace=False)
    sell_count = dataset.user_id.value_counts()
    sell = pd.DataFrame(sell_count)
    sell = sell.rename(columns= {'user_id':'sell_count'})
    sell = sell.reset_index()
    sell.columns = ['user_id','sell_count']
    return sell.loc[sell.sell_count >=13].loc[sell.sell_count <= 100].user_id.unique()

def generate_dataset(user_set, if_test, purchase_list, len_q, purchase_list_2 = None):
    np.random.seed(6298)
    support_set_x = []
    support_set_y = []
    query_set_x = []
    query_set_y = []
    

    if if_test == 0:
        condidate_ngtv_set = []
        for items in purchase_list.values():
            condidate_ngtv_set.extend(items)
    
    if if_test == 1:
        condidate_ngtv_set = []
        for items in purchase_list_2.values():
            condidate_ngtv_set.extend(items)

    
    n = 0
    for user in user_set:
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        purchased_item = purchase_list[user]
        #if len(purchased_item) > 20:
        #    purchased_item = purchased_item[-20:]
        support_item = purchased_item[:-len_q]
        iteration_length = len(purchased_item)-len_q
        query_item = purchased_item[-len_q:]
         
        neg_item = list(set(condidate_ngtv_set))
        if if_test == 1:
            try:
                neg_item = np.setdiff1d(neg_item, purchase_list_2[user])
            except:
                neg_item = neg_item
        for i in range(iteration_length):  
            ngtv_index = np.random.randint(0,len(neg_item),size=1)  
            while neg_item[ngtv_index[0]] in purchase_list[user]:
                ngtv_index = np.random.randint(0,len(neg_item),size=1)
            support_x.append([user, support_item[i]])
            support_y.append([1])
            support_x.append([user, neg_item[ngtv_index[0]]])
            support_y.append([0])
        
        for i in range(len_q): 
            ngtv_index = np.random.randint(0,len(neg_item),size=1)  
            while neg_item[ngtv_index[0]] in purchase_list[user]:
                ngtv_index = np.random.randint(0,len(neg_item),size=1)
            query_x.append([user, query_item[i]])
            query_y.append([1])
            query_x.append([user, neg_item[ngtv_index[0]]])
            query_y.append([0])
        
        
        
        support_set_x.append(support_x)
        support_set_y.append(support_y)
        query_set_x.append(query_x)
        query_set_y.append(query_y)
        
        
    return support_set_x, support_set_y, query_set_x, query_set_y

def prepare_dataset_MeLU():
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
    support_set_x, support_set_y, query_set_x, query_set_y = generate_dataset(Tr_user, 0, Tr_rated_dict, 5)
    total_dataset = list(zip(support_set_x, support_set_y, query_set_x, query_set_y))
    valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y = generate_dataset(V_user, 1, V_rated_dict, 5, Tr_rated_dict)
    valid_dataset = list(zip(valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y))
    return total_dataset, valid_dataset, train_order, valid_order, test_order, \
            movie_dict, user_dict, Tr_rated_dict, Rated_dict, condidate_item

class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']

        self.item_emb = item(config)
        self.user_emb = user(config)
              
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.d_1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.d_2 = torch.nn.Dropout(0.5)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training = True):
        
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        x = self.d_1(F.relu(x))
        x = self.fc2(x)
        x =self.d_2(F.relu(x))
        #x = self.fc3(x)
        #x = F.relu(x)
        x = self.linear_out(x)
        x = F.sigmoid(x)
        return x
    
class MeLU(torch.nn.Module):
    def __init__(self, config, movie_dict, user_dict):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']
        
        self.movie_dict = movie_dict
        self.user_dict = user_dict

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.pretrain_per_mini_batch(support_set_x)
            #loss = F.mse_loss(support_set_y_pred, torch.Tensor(support_set_y))
            #print(support_set_y_pred)
            #print(torch.LongTensor(support_set_y))
            loss = F.binary_cross_entropy(support_set_y_pred, torch.Tensor(support_set_y))
            #print(loss)

            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.pretrain_per_mini_batch(query_set_x)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []

        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
                
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            #loss_q = F.mse_loss(query_set_y_pred, torch.Tensor(query_set_ys[i]))
            loss_q = F.binary_cross_entropy(query_set_y_pred, torch.Tensor(query_set_ys[i]))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return losses_q.data
    
    def pretrain_per_mini_batch(self, x):
        pstv_input = torch.LongTensor(len(x), 10246)
        i = 0

        for u, p_v in x:
            pstv_input[i,:10242] = self.movie_dict[p_v]
            pstv_input[i,10242:] = self.user_dict[u]
            i += 1  
        ps_rslt = self.model(pstv_input)
        return ps_rslt

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        tmp = 0.

        if self.cuda():
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()

        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.pretrain_per_mini_batch(support_set_x)
            #loss = F.mse_loss(support_set_y_pred, torch.Tensor(support_set_y))
            loss = F.binary_cross_entropy(support_set_y_pred, torch.Tensor(support_set_y))
            # unit loss
            loss /= torch.norm(loss).tolist()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for i in range(self.weight_len):
                # For averaging Forbenius norm.
                tmp += torch.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / num_local_update

def MeLU_training(melu, total_dataset, valid_dataset, batch_size, num_epoch, model_save=True, model_filename=None):

    if configs_MeLU['use_cuda']:
        melu.cuda()

    valid_s_set_x, valid_s_set_y, valid_q_set_x, valid_q_set_y = zip(*valid_dataset)

    batch_sz = len(valid_s_set_x)    
    training_set_size = len(total_dataset)
    min_valid_q_loss = 999999
    for epoch in range(num_epoch):
        melu.train()
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        losses_q = []
        for i in range(num_batch):
            starttime = time.time()
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            losses_q_batch = melu.global_update(supp_xs, supp_ys, query_xs, query_ys, configs_MeLU['inner'])
            endtime = time.time()
            
            sys.stdout.write("\r epoch:{0}  {1}/{2}  batch_loss_query_set: {3} time: {4}"
                             .format(epoch, i, num_batch, losses_q_batch, endtime - starttime))
            sys.stdout.flush()  
            losses_q.append(losses_q_batch)
        losses_q = torch.stack(losses_q).mean(0)
        #melu.eval()     
        
        valid_losses_q = []

        for i in range(batch_sz):
            valid_q_y_pred = melu.forward(valid_s_set_x[i], valid_s_set_y[i], valid_q_set_x[i], configs_MeLU['inner'])
            #valid_loss_q = F.mse_loss(valid_q_y_pred, torch.Tensor(valid_q_set_y[i]))
            valid_loss_q = F.binary_cross_entropy(valid_q_y_pred, torch.Tensor(valid_q_set_y[i]))
            valid_losses_q.append(valid_loss_q)
        valid_losses_q = torch.stack(valid_losses_q).mean(0)
        
        
        print("")
        print('epoch_' + str(epoch) + ":  epoch loss_query_set = " + str(losses_q.data) + "  valid_loss = " + str(valid_losses_q))
        if min_valid_q_loss > valid_losses_q:
            min_valid_q_loss = valid_losses_q
            torch.save(melu, "MeLU")
            patient = 10
            continue
        if min_valid_q_loss <= valid_losses_q:
            patient = patient - 1
        if patient == 0:
            break

def MeLU_generate_dataset_for_Rec(test_order, purchase_list, bias, duration, min_pchs, len_query_i, f_w_u):
    np.random.seed(52)
    sup_init_day = test_order.days.min() + bias
    
    query_users = test_order.loc[test_order.days > (sup_init_day + duration//2)]\
    .loc[test_order.days <= (sup_init_day + duration)]
    
    query_item_list = query_users.movie_id.unique()        
    
    s_x, s_y, Rec_user, q_item_list, Dict_T_purchuse_list = \
    MeLU_sample_sub_s_q_4_rec(query_users, purchase_list, query_item_list, 
                              bias, min_pchs, len_query_i, f_w_u)
    
    return s_x, s_y, Rec_user, q_item_list, Dict_T_purchuse_list

def MeLU_sample_sub_s_q_4_rec(df_data, purchase_list, item_list, bias, min_pchs, len_query_i, f_w_u):
    sub_support_set_x = []
    sub_support_set_y = []    
    q_item_list = df_data.movie_id.unique()
    
    GndThu = np.load("./Dataset/Ground_truth/Ground_truth_bias{0}_wu{1}.npy".format(bias, f_w_u), allow_pickle=True).item()
    user_set = list(GndThu.keys())

    Dict_T_purchuse_list = {}   
    for u in user_set:
        u_support_x = []
        u_support_y = []

        purchased_item = df_data.loc[df_data.user_id == u].movie_id.values
        #print(len(GndThu[u]), len(purchased_item))
        neg_item = item_list
    
        for i in range(len(purchased_item) - len_query_i):
            ngtv_index = np.random.randint(0,len(neg_item),size=1)  
            try:
                while neg_item[ngtv_index[0]] in purchase_list[u]:
                    ngtv_index = np.random.randint(0,len(neg_item),size=1)
            except:
                ngtv_index = np.random.randint(0,len(neg_item),size=1) 

            u_support_x.append([u, purchased_item[i]])
            u_support_y.append([1])
            u_support_x.append([u, neg_item[ngtv_index[0]]])
            u_support_y.append([0])
        
        Dict_T_purchuse_list[u] = purchased_item[-len_query_i:]

        sub_support_set_x.append(u_support_x)
        sub_support_set_y.append(u_support_y)
    return sub_support_set_x, sub_support_set_y, user_set, q_item_list, Dict_T_purchuse_list

def MeLU_Recommending(test_order, Tr_rated_dict, Rated_dict, condidate_item):
    biases = [0, 14, 28]
    w_u = [0, 1]
    for f_w_u in w_u:
        for bias in biases:
            print("--------------------------------------------------------")
            print("bias_" + str(bias) + " start")
            test_initial_day = test_order.days.head(1).values[0]
                                                                                
            q_sub_s_x, q_sub_s_y, Rec_user, q_item_list, Dict_T_purchuse_list\
            = MeLU_generate_dataset_for_Rec(test_order, Tr_rated_dict, bias, configs_MeLU['durations'], 
                                            configs_MeLU['min_pchs'], configs_MeLU['len_query_i'], f_w_u)
            try:
                model = torch.load("MeLU")
            except:
                model = melu
            pred_reclist = {} 
            for i in range(len(Rec_user)):
                q_set_q = []
                condidate_item_user = np.setdiff1d(condidate_item, Rated_dict[Rec_user[i]])
                condidate_item_user = np.union1d(condidate_item_user, Dict_T_purchuse_list[Rec_user[i]])
                for j in range(len(condidate_item_user)):
                    q_set_q.append([Rec_user[i], condidate_item_user[j]]) 
                pred_y = model.forward(q_sub_s_x[i], q_sub_s_y[i], q_set_q, 1)
                pred_score = dict(zip(condidate_item_user, pred_y.data))
                pred_score = sorted(pred_score.items(),key = lambda x:x[1],reverse = True)
                pred_score_top100 = pred_score[:100]
                for k in range(100):
                    pred_score_top100[k] = pred_score_top100[k][0]  
                pred_reclist[Rec_user[i]] = pred_score_top100
            np.save("./Rec_result/MeLU_Reclist_bias{0}_wu{1}.npy".format(bias, f_w_u), pred_reclist, allow_pickle=True)
            print("bias_" + str(bias) + " end")
            print("--------------------------------------------------------")
