import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_reclist(direction, document):
    dir_list = [i for i in os.listdir(direction) if i.startswith(document)]
    dict_Reclist = {}
    for i in range(len(dir_list)):
        dict_Reclist.update(np.load(direction + dir_list[i], allow_pickle=True).item())
    return dict_Reclist


def Evaluation(Reclist, user_bought_testset, NumInter):
    hit_rite = []
    #recall = []
    ndcg = []
    for i in range(100):
        hr_i = 0.
        recall_i = 0.
        ndcg_i = 0.
        for user in Reclist.keys():
            hr_i += len(set(Reclist[user][:i+1]) & set(user_bought_testset[user][-3:]))
            #recall_i += len(set(Reclist[user][:i+1]) & set(user_bought_testset[user][-3:])) / len(user_bought_testset[user][-3:])
            ndcg_i += getNDCG(Reclist[user][:i+1], user_bought_testset[user][-3:])
        #recall.append(recall_i/len(Reclist))
        hit_rite.append(hr_i / NumInter)
        ndcg.append(ndcg_i / NumInter)
    return hit_rite, ndcg


def Generate_result(direction, document, testset, NumInter): 
    dict_Reclist = load_reclist(direction, document)
    hit_rate, ndcg = Evaluation(dict_Reclist, testset, NumInter)
    return hit_rate, ndcg

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def calculate(b, w_u):
    Ground_truth = np.load("./Dataset/Ground_truth/Ground_truth_bias{0}_wu{1}.npy".format(b, w_u), allow_pickle=True).item()
    N_intrctn = 0
    for items in Ground_truth.values():
        N_intrctn += len(items)

    hr, ndcg = Generate_result('./Rec_result/', "Reclist_bias{0}_wu{1}.npy".format(b, w_u), Ground_truth, N_intrctn)
    hr_1, ndcg_1 = Generate_result('./Rec_result/', "MeLU_Reclist_bias{0}_wu{1}.npy".format(b, w_u), Ground_truth, N_intrctn)
    hr_2, ndcg_2 = Generate_result('./Rec_result/', "MAMO_Reclist_bias{0}_wu{1}.npy".format(b, w_u), Ground_truth, N_intrctn)
    return hr, hr_1, hr_2, ndcg, ndcg_1, ndcg_2

def Show_result():
    a = [1]
    a.extend([(i+1)*10 for i in range(10)])
    bias_list = [0,14,28]
    w_u = [0, 1]
    colors_ = ["r", "b", "g"]

    #hit_rate_list = []
    #recall_list = []
    for f_w_u in w_u:
        hit_ave = None
        hit_ave_1 = None
        hit_ave_2 = None
        NDCG_ave = None
        NDCG_ave_1 = None
        NDCG_ave_2 = None
        for bias in bias_list:
            try:
                hit_rate, hit_rate_1, hit_rate_2, ndcg, ndcg_1, ndcg_2 = calculate(bias, f_w_u)
                try:
                    hit_ave += np.array(hit_rate)
                    hit_ave_1 += np.array(hit_rate_1)
                    hit_ave_2 += np.array(hit_rate_2)

                except:
                    hit_ave = np.array(hit_rate)
                    hit_ave_1 = np.array(hit_rate_1)
                    hit_ave_2 = np.array(hit_rate_2)

                try:
                    NDCG_ave += np.array(ndcg)
                    NDCG_ave_1 += np.array(ndcg_1)
                    NDCG_ave_2 += np.array(ndcg_2)

                except:
                    NDCG_ave = np.array(ndcg)
                    NDCG_ave_1 = np.array(ndcg_1)
                    NDCG_ave_2 = np.array(ndcg_2)              
            except:
                continue
        hit_ave = np.array(hit_ave)/3
        hit_ave_1 = np.array(hit_ave_1)/3
        hit_ave_2 = np.array(hit_ave_2)/3

        NDCG_ave = np.array(NDCG_ave)/3
        NDCG_ave_1 = np.array(NDCG_ave_1)/3
        NDCG_ave_2 = np.array(NDCG_ave_2)/3    

        plt.plot([i+1 for i in range(100)], hit_ave, label = "HML", color="r")
        plt.plot([i+1 for i in range(100)], hit_ave_1, label = "MeLU", color="b")
        plt.plot([i+1 for i in range(100)], hit_ave_2, label = "MAMO", color="g")
        plt.xlabel('K')
        plt.ylabel('# HR@K')
        plt.xticks(a)
        plt.grid()
        plt.legend()
        if f_w_u == 0:
            plt.savefig('./Figures/HR@K_cold.pdf'.format(f_w_u))
        if f_w_u == 1:
            plt.savefig('./Figures/HR@K_warm.pdf'.format(f_w_u))
        plt.show()
        plt.plot([i+1 for i in range(100)], NDCG_ave, label = "HML", color="r")
        plt.plot([i+1 for i in range(100)], NDCG_ave_1, label = "MeLU", color="b")
        plt.plot([i+1 for i in range(100)], NDCG_ave_2, label = "MAMO", color="g")
        plt.xlabel('K')
        plt.ylabel('# NDCG@K')
        plt.xticks(a)
        plt.grid()
        plt.legend()
        if f_w_u == 0:
            plt.savefig('./Figures/NDCG@K_cold.pdf'.format(f_w_u))
        if f_w_u == 1:
            plt.savefig('./Figures/NDCG@K_warm.pdf'.format(f_w_u))
        plt.show()