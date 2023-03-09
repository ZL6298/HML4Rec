configs = {
    # item side information
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'embedding_dim': 32,
    # user side information
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    #hyper-parameters
    'safety_margin_size' : 0.2,
    'iterations' : 1000,
    'durations' : 14,
    'global_lr' : 1e-3,
    'local_lr' : 5e-5,
    'market_lr' : 5e-4,
    'ways_t' : 16, # periods number per mini-batch
    'ways_s' : 20, # users number per period
    'min_pchs' : 6,
    'len_query_i' : 3,
}

configs_MeLU = {
    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'embedding_dim': 16,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 16,
    # cuda setting
    'use_cuda': False,
    # model setting
    'inner': 1,
    'lr': 5e-5,
    'local_lr': 5e-6,
    'batch_size': 32,
    'num_epoch': 50,
    'min_pchs' : 6,
    'len_query_i' : 3,
    'durations' : 14,
}

config_MAMO = {
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'rand': True,
    'random_state': 100,
    'split_ratio': 0.8,
    'support_size': 15,
    'query_size': 5,
    'embedding_dim': 64,
    'n_layer': 2,
    'alpha': 0.5,
    'beta': 0.05,
    'gamma': 0.01,
    'rho': 0.001,
    'lamda': 0.005,
    'tao': 0.01,
    'n_k': 3,
    'batch_size': 10,
    'n_epoch': 3,
    'n_inner_loop': 2,
    'active_func': 'leaky_relu',
    'cuda_option': 'cuda:5'
}
