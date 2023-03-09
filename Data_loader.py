import numpy as np 
import pandas as pd 
import datetime
import random 
import torch 
import time
from scipy.stats import entropy
import re

def load_movielens_1m():
    path = "./Dataset"
    profile_data_path = "{}/users.dat".format(path)
    score_data_path = "{}/ratings.dat".format(path)
    item_data_path = "{}/movies_extrainfos.dat".format(path) 

    profile_data = pd.read_csv(
        profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'], 
        sep="::", engine='python'
    )
    item_data = pd.read_csv(
        item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'], 
        sep="::", engine='python', encoding="utf-8"
    )
    score_data = pd.read_csv(
        score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
        sep="::", engine='python'
    )
    score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x)) 
    return profile_data, item_data, score_data

def prepare_dataset():
    profile_data, item_data, score_data = load_movielens_1m()
    #==============================#
    #       2000,04 - 2001,01      # 
    #==============================#
    score_data.sort_values("time", inplace=True)
    score_data["date"] = score_data.time.dt.date
    score_data = score_data.loc[score_data.date < datetime.date(2001,1,1)]
    initial_date = score_data.date.head(1).values[0]
    score_data['days']  = score_data["date"].map(lambda x: (x - initial_date).days)
    print("Dataset:")
    print("# of users: {0}".format(len(score_data.user_id.unique())))
    print("# of movies: {0}".format(len(score_data.user_id.unique())))
    print("# of interactions: {0}".format(score_data.shape[0]))
          
    train_order = score_data.loc[score_data.date < datetime.date(2000,11,1)]
    valid_order = score_data.loc[score_data.date < datetime.date(2000,11,15)].loc[score_data.date >= datetime.date(2000,11,1)]
    test_order = score_data.loc[score_data.date >= datetime.date(2000,11,15)]
    
    train_order = train_order[['user_id', 'movie_id', 'days', 'rating']]
    valid_order = valid_order[['user_id', 'movie_id', 'days', 'rating']]
    test_order = test_order[['user_id', 'movie_id', 'days', 'rating']]
    
    movie_dict, user_dict = build_feature_dict(profile_data, item_data)
    Tr_rated_dict = build_rated_dict(train_order)
    Rated_dict = build_rated_dict(score_data)
    condidate_item = score_data.movie_id.unique()
    condidate_item = np.random.choice(condidate_item, 1500, replace=False)
    
    return train_order, valid_order, test_order, movie_dict, user_dict, Tr_rated_dict, Rated_dict, condidate_item

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)

def user_converting(row, gender_list, age_list, occupation_list, zipcode_list): 
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)

def build_feature_dict(profile_data, item_data, dataset_path = "./Dataset"):
    # load content feature list
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))
    
    movie_dict = {}
    for idx, row in item_data.iterrows():
        m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
        movie_dict[row['movie_id']] = m_info
        
    user_dict = {}
    for idx, row in profile_data.iterrows():
        u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
        user_dict[row['user_id']] = u_info
    return movie_dict, user_dict

def build_rated_dict(train_order):
    Tr_rated_dict = {}
    for users in train_order.user_id.unique():
        bought_list = train_order.loc[train_order.user_id == users].movie_id.values
        Tr_rated_dict[users] = bought_list
    return Tr_rated_dict

def generate_dataset_per_task(train_order, valid_order, purchase_list, duration, ways_s, way_q, min_pchs, len_query_i, training):
    if training:
        sup_init_day = np.random.choice(np.arange(train_order.days.max()+1-duration + 1),1)[0]
        support_user = train_order.loc[train_order.days >= sup_init_day].loc[train_order.days < (sup_init_day + duration//2)]
        query_user = train_order.loc[train_order.days >= (sup_init_day + duration//2)].loc[train_order.days < (sup_init_day + duration)]
    else:
        sup_init_day = valid_order.days.min()
        support_user = valid_order.loc[valid_order.days >= sup_init_day].loc[valid_order.days < (sup_init_day + duration//2)]
        query_user = valid_order.loc[valid_order.days >= (sup_init_day + duration//2)].loc[valid_order.days < (sup_init_day + duration)]

    sup_item_list = support_user.movie_id.unique()
    query_item_list = query_user.movie_id.unique()
    
    u_s_s, u_s_q, s_market_entropy = sample_interactions(support_user, purchase_list, sup_item_list, ways_s, min_pchs, len_query_i)
    
    u_q_s, u_q_q, q_market_entropy = sample_interactions(query_user, purchase_list, query_item_list, way_q, min_pchs, len_query_i)
        
    return u_s_s, u_s_q, u_q_s, u_q_q, s_market_entropy

def sample_interactions(df_data, purchase_list, item_list, ways_s, min_pchs, len_query_i):
    sub_support_set_x = []
    sub_query_set_x = []
    
    df_new = pd.DataFrame(df_data.user_id.value_counts())
    df_new = df_new.reset_index()
    df_new.columns = ["user_id", "sale_amount"]
    user_set = df_new.loc[df_new.sale_amount >= min_pchs].user_id.unique()
    
    
    user_set = np.random.choice(user_set, ways_s)
    market_dist = []
    for u in user_set:
        u_support_x = []
        u_query_x = []

        purchased_item = df_data.loc[df_data.user_id == u].movie_id.values
        neg_item = item_list
        
        for i in range(len(purchased_item) - len_query_i):
            ngtv_index = np.random.randint(0,len(neg_item),size=1)  
            try:
                while neg_item[ngtv_index[0]] in purchase_list[u]:
                    ngtv_index = np.random.randint(0,len(neg_item),size=1)
            except:
                ngtv_index = np.random.randint(0,len(neg_item),size=1) 

            u_support_x.append([u, purchased_item[i], neg_item[ngtv_index[0]]])
            
        for i in range(len_query_i):
            ngtv_index = np.random.randint(0,len(neg_item),size=1)  
            try:
                while neg_item[ngtv_index[0]] in purchase_list[u]:
                    ngtv_index = np.random.randint(0,len(neg_item),size=1)
            except:
                ngtv_index = np.random.randint(0,len(neg_item),size=1) 
                
            u_query_x.append([u, purchased_item[-(i+1)], neg_item[ngtv_index[0]]]) 

        sub_support_set_x.append(u_support_x)
        sub_query_set_x.append(u_query_x)
        
        market_dist.extend(purchased_item)
    market_entropy = entropy(pd.value_counts(market_dist).values / len(market_dist))
    return sub_support_set_x, sub_query_set_x, market_entropy

def generate_dataset_for_Rec(train_order, test_order, purchase_list, duration, ways_s, bias, f_w_u, min_pchs, len_query_i):
    np.random.seed(52)
    sup_init_day = test_order.days.min() + bias
    support_users = test_order.loc[test_order.days > sup_init_day].loc[test_order.days <= (sup_init_day + duration//2)]
    query_users = test_order.loc[test_order.days > (sup_init_day + duration//2)].loc[test_order.days <= (sup_init_day + duration)]

    sup_item_list = support_users.movie_id.unique()
    query_item_list = query_users.movie_id.unique()
    
    u_s_s, u_s_q, s_market_entropy = sample_interactions(support_users, purchase_list, 
                                                         sup_item_list, ways_s, min_pchs, len_query_i)
    
    u_q_s, Rec_user, q_item_list, Dict_T_purchuse_list = sample_interactions_for_rec(train_order, query_users, purchase_list, 
                                                                                       query_item_list, f_w_u, 
                                                                                       min_pchs, len_query_i)
        
    return u_s_s, u_s_q, u_q_s, Rec_user, q_item_list, Dict_T_purchuse_list, s_market_entropy

def sample_interactions_for_rec(train_order, df_data, purchase_list, item_list, f_w_u, min_pchs, len_query_i):
    q_item_list = df_data.movie_id.unique()
    
    sub_support_set_x = []

    df_new = pd.DataFrame(df_data.user_id.value_counts())
    df_new = df_new.reset_index()
    df_new.columns = ["user_id", "sale_amount"]
    user_set = df_new.loc[df_new.sale_amount >= min_pchs].user_id.unique()
    
    Tr_user_list = train_order.user_id.unique()
    if f_w_u == 1:
        user_set = np.intersect1d(user_set, Tr_user_list)
    if f_w_u == 0:
        user_set = np.setdiff1d(user_set, Tr_user_list)
    
    Tr_item_list = train_order.movie_id.unique()
    
    neg_item = item_list
    Dict_T_purchuse_list = {}   
    for u in user_set:
        u_support_x = []
        u_query_x = []

        purchased_item = df_data.loc[df_data.user_id == u].movie_id.values
        
        
        for i in range(len(purchased_item) - len_query_i):
            ngtv_index = np.random.randint(0,len(neg_item),size=1)  
            try:
                while neg_item[ngtv_index[0]] in purchase_list[u]:
                    ngtv_index = np.random.randint(0,len(neg_item),size=1)
            except:
                ngtv_index = np.random.randint(0,len(neg_item),size=1) 

            u_support_x.append([u, purchased_item[i], neg_item[ngtv_index[0]]])
        
        Dict_T_purchuse_list[u] = purchased_item[-len_query_i:]

        sub_support_set_x.append(u_support_x)
    return sub_support_set_x, user_set, q_item_list, Dict_T_purchuse_list
