from munch import Munch

#from .data.prepdata import CLS, SEP, PAD, MASK
# according to the csv, max(tokens) = 30005
# while, they are sparse (len(set(tokens)) == ~8000, not 30005), so consider constracting it if need more expansion of scale in memory


## this is for experiment configuration
EXPCONF = {
    #debug option
    'debug':False,

	'tsz': 30000,
	'dsz': 40000,
	'train_mode': True,
	'eval_mode' : False,

    #'use_pretrained': False,
	'vocab_size': 30010,
    'embedding_dim': 5,
	'input_size': 5,
	'hidden_size': 128,
	'num_layers': 1,
	# 'num_hidden_layers': 8,
	# 'num_attention_heads': 8,
	
	'seed': 777,
    # datapath and dataloader  == loading f{dataroot}/{mode}{kfold_k}.jsonl
    'numworkers': 0, #hard to tell what is optimal... but consider number of cpus we have https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5

    'see_bsz_effect': False, #with this option =True, logs are recorded with x = number of examples seen
                            # this is just confusing...

    # training conditions
    'numep': 10, # later optimize
	'bsz': 32,
    'scheduler': 'cosine', # linear
    'warmups': 100,
    'lr': 1e-4,
    'modelsaveroot': 'model/', #path to save .pth
    # PP loss balancing coeff  alpha_pp
    'alpha_pp': 0.5, # float
        'alpha_warmup':False, # if True, it grows from 0 to alpha_pp according to warmup_steps

    #adamW
    'weight_decay': 0,


    # experiment condition
    'maskratio': 0.15,
    'masking': 'random', # span (span masking used for ALBERT original paper )
        'span_n': 3, # to what n-gram would span masking cover
    'savethld':0.45,

}

EXPCONF = Munch(EXPCONF)
