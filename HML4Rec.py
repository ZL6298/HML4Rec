import numpy as np 
import pandas as pd 
import random 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from configs import *
from Data_loader import *
from models import *

class HML4Rec:
    def __init__(self, configs, train_order, valid_order, test_order, 
                 movie_dict, user_dict, Tr_rated_dict, Rated_dict, condidate_item):
        self.iterations = configs["iterations"]
        self.durations = configs["durations"]
        self.ways_t = configs["ways_t"]
        self.ways_s = configs["ways_s"]
        self.min_pchs = configs["min_pchs"]
        self.len_query_i = configs["len_query_i"]
        self.local_lr = configs["local_lr"]
        self.market_lr = configs["market_lr"]
        self.global_lr = configs["global_lr"]
        self.safety_margin_size = configs["safety_margin_size"]
        
        self.train_order = train_order
        self.valid_order = valid_order
        self.test_order = test_order
        self.movie_dict = movie_dict
        self.user_dict = user_dict
        self.Tr_rated_dict = Tr_rated_dict
        self.Rated_dict = Rated_dict
        self.condidate_item = condidate_item
        
        
        self.model = Rec_Model(configs)
        self.optimizer = Adam(self.model.parameters(), lr=self.global_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.95)
        
        
    
    def hinge_loss(self, ps_rslt, neg_rslt, safety_margin_size):
        return F.relu(safety_margin_size - ps_rslt + neg_rslt)

    def pretrain_per_mini_batch(self, x, model):
    
        pstv_input = torch.LongTensor(len(x), 10246)
        ngtv_input = torch.LongTensor(len(x), 10246)
        i = 0

        for u, p_v, n_v in x:
            pstv_input[i,:10242] = self.movie_dict[p_v]
            pstv_input[i,10242:] = self.user_dict[u]
            ngtv_input[i,:10242] = self.movie_dict[n_v]
            ngtv_input[i,10242:] = self.user_dict[u]
            i += 1

        neg_rslt = model(ngtv_input)
        ps_rslt = model(pstv_input)
        return ps_rslt, neg_rslt

    def local_training(self, model, support_set, query_set):
    
        weight_for_local_update = list(model.state_dict().values())
        ps_rslt, neg_rslt = self.pretrain_per_mini_batch(support_set, model)
        loss = self.hinge_loss(ps_rslt, neg_rslt, self.safety_margin_size)
        batch_loss = torch.sum(loss)
        model.zero_grad()
        grad = torch.autograd.grad(batch_loss, model.parameters(), create_graph=True)   
        for i in range(model.weight_len):
            if model.weight_name[i] in model.local_update_target_weight_name:
                model.fast_weights[model.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
            else:
                model.fast_weights[model.weight_name[i]] = weight_for_local_update[i]
        model.load_state_dict(model.fast_weights)
        ps_rslt, neg_rslt = self.pretrain_per_mini_batch(query_set, model)
        model.load_state_dict(model.keep_weight)

        return ps_rslt, neg_rslt, batch_loss.data / len(support_set)

    def market_adapting(self, sub_s, sub_q, ways_s, model, s_market_entropy):

        batch_Q_loss = []    
        batch_S_loss = []
        weight_for_market_update = list(model.state_dict().values())
        for i in range(ways_s):
            model.store_parameters()
            model.train()
            q_ps, q_neg, S_loss = self.local_training(model, sub_s[i], sub_q[i])
            batch_S_loss.append(S_loss)

            loss = self.hinge_loss(q_ps, q_neg, self.safety_margin_size)
            query_loss = torch.sum(loss)
            batch_Q_loss.append(query_loss)

        batch_Q_loss = torch.stack(batch_Q_loss).mean(0)
        model.zero_grad()
        grad = torch.autograd.grad(batch_Q_loss, model.parameters(), create_graph=True)
        for i in range(model.weight_len):
            if model.weight_name[i] in model.market_update_target_weight_name:
                model.fast_weights[model.weight_name[i]] = weight_for_market_update[i] - s_market_entropy * self.market_lr * grad[i]
            else:
                model.fast_weights[model.weight_name[i]] = weight_for_market_update[i]
        model.load_state_dict(model.fast_weights)
        model.store_parameters()
        return np.array(batch_S_loss).mean(), batch_Q_loss.data
    
    def global_updating(self, train_order, valid_order, duration, ways_t, ways_s, model, min_pchs, len_query_i, training):
        loss_s_top_q = []
        loss_q_top_q = []
        model.store_top_parameters()
        for i in range(ways_t):
            s_sub_s, s_sub_q, q_sub_s, q_sub_q, s_market_entropy = generate_dataset_per_task(train_order, valid_order, self.Tr_rated_dict, duration, ways_s, ways_s, min_pchs, len_query_i, training)
            _, s_sub_q_loss = self.market_adapting(s_sub_s, s_sub_q, ways_s, model, s_market_entropy) #只有P_m更新过一次
            loss_s_top_q.append(s_sub_q_loss)
            for i in range(len(q_sub_s)):
                q_ps, q_neg, _ = self.local_training(model, q_sub_s[i], q_sub_q[i]) #q_q计算完后，p_m没有重置
                loss = self.hinge_loss(q_ps, q_neg, self.safety_margin_size)
                query_loss = torch.sum(loss)
                loss_q_top_q.append(query_loss)
            model.load_state_dict(model.keep_weights_top)
        loss_q_top_q = torch.stack(loss_q_top_q).mean(0)
        if training:
            model.zero_grad()
            loss_q_top_q.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            self.optimizer.step()
            model.store_parameters()

        return np.array(loss_s_top_q).mean()/len_query_i, loss_q_top_q.data/len_query_i
    def train(self):
        train_Sq_loss, train_Qq_loss = [], []
        valid_Sq_loss, valid_Qq_loss = [], []
        min_valid_Qq_loss = 999999
        np.random.seed(6298)
        for episode in range(self.iterations):
            t_s = time.time()
            #Training
            e_tr_s_loss, e_tr_q_loss = self.global_updating(self.train_order, self.valid_order, self.durations, self.ways_t, self.ways_s, 
                                                       self.model, self.min_pchs, self.len_query_i, 1)
            train_Sq_loss.append(e_tr_s_loss)
            train_Qq_loss.append(e_tr_q_loss)
            #Validation
            e_v_s_loss, e_v_q_loss = self.global_updating(self.train_order, self.valid_order, self.durations, self.ways_t, self.ways_s, 
                                                       self.model, self.min_pchs, self.len_query_i, 0)
            valid_Sq_loss.append(e_v_s_loss)
            valid_Qq_loss.append(e_v_q_loss)
            t_e = time.time()
            if episode % 10 == 0:
                print("Episode: {0}  tr_S_loss: {1} tr_Q_loss: {2}".format(episode, e_tr_s_loss, e_tr_q_loss), end=" ")
                print("v_S_loss: {0}  v_Q_loss: {1}  time: {2}".format(e_v_s_loss, e_v_q_loss, round(t_e-t_s),4))


            if e_v_q_loss < min_valid_Qq_loss:
                torch.save(self.model, 'HML4Rec')
                min_valid_Qq_loss = e_v_q_loss
                patient = 40
                continue
            if e_v_q_loss >= min_valid_Qq_loss:
                self.scheduler.step(e_v_q_loss)
                patient = patient - 1
            if patient == 0:
                break

        print("")
        print("min_Eloss: " + str(min_valid_Qq_loss))
        print("----------------------------------------------")   
        
    def prediction(self, x, model):
        pstv_input = torch.LongTensor(len(x), 10246)

        i = 0
        for u, p_v in x:
            pstv_input[i,:10242] = self.movie_dict[p_v]
            pstv_input[i,10242:] = self.user_dict[u]
            i += 1
        ps_rslt = model(pstv_input)

        return ps_rslt
        
    def Recommending(self):
        bias_list = [0, 0+self.durations, 0+2*self.durations]
        w_u = [0, 1]
        for f_w_u in w_u:
            for bias in bias_list:
                print("--------------------------------------------------------")
                print("bias_" + str(bias) + " start!")

                sub_s, sub_q, q_sub_s, Rec_user, q_item_list, T_rated_dict, s_market_entropy = \
                generate_dataset_for_Rec(self.train_order, self.test_order, self.Tr_rated_dict, 
                                         self.durations, self.ways_s, bias, f_w_u, self.min_pchs, self.len_query_i)
                try:
                    T_rated_dict = np.load("./Dataset/Ground_truth_bias{0}_wu{1}.npy".format(bias, f_w_u), allow_pickle=True).item()
                except:
                    np.save("./Dataset/Ground_truth_bias{0}_wu{1}.npy".format(bias, f_w_u), T_rated_dict)
                
                try:
                    model = torch.load("HML4Rec")
                except:
                    model = self.model
                    
                _, s_sub_q_loss = self.market_adapting(sub_s, sub_q, self.ways_s, model, s_market_entropy) #只有P_m更新过一次

                pred_reclist = {}
                for i in range(len(Rec_user)):

                    weight_for_local_update = list(model.state_dict().values())
                    ps_rslt, neg_rslt = self.pretrain_per_mini_batch(q_sub_s[i], model)
                    loss = self.hinge_loss(ps_rslt, neg_rslt, self.safety_margin_size)
                    batch_loss = torch.sum(loss)
                    model.zero_grad()
                    grad = torch.autograd.grad(batch_loss, model.parameters(), create_graph=True)   
                    for j in range(model.weight_len):
                        if model.weight_name[j] in model.local_update_target_weight_name:
                            model.fast_weights[model.weight_name[j]] = weight_for_local_update[j] - self.local_lr * grad[j]
                        else:
                            model.fast_weights[model.weight_name[j]] = weight_for_local_update[j]
                    model.load_state_dict(model.fast_weights)
                    q_set_q = []
                    condidate_item_user = np.setdiff1d(self.condidate_item, self.Rated_dict[Rec_user[i]])
                    condidate_item_user = np.union1d(condidate_item_user, T_rated_dict[Rec_user[i]])
                    for j in range(len(condidate_item_user)):
                        q_set_q.append([Rec_user[i], condidate_item_user[j]]) 
                    pred_rslt = self.prediction(q_set_q, model)
                    model.load_state_dict(model.keep_weight)

                    pred_score = dict(zip(condidate_item_user, pred_rslt.data))
                    pred_score = sorted(pred_score.items(),key = lambda x:x[1],reverse = True)

                    pred_score_top100 = pred_score[:100]
                    for k in range(100):
                        pred_score_top100[k] = pred_score_top100[k][0]

                    pred_reclist[Rec_user[i]] = pred_score_top100

                np.save("./Rec_result/Reclist_bias{0}_wu{1}.npy".format(bias, f_w_u), pred_reclist, allow_pickle=True)
                print("Results store at: ./Rec_result/Reclist_bias{0}_wu{1}.npy".format(bias, f_w_u))
                print("bias_" + str(bias) + " end")
                print("--------------------------------------------------------")