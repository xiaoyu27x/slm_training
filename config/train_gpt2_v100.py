wandb_log = False
dataset = 'owt_20k'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

max_iters = 1000
lr_decay_iters = 1000
warmup_iters = 100

eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

device = 'cuda'
dtype = 'float16'
compile = False
out_dir = 'out-owt-20k'
