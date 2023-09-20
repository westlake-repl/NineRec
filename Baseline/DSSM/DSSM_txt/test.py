import os

gpu_device = '0_132'

root_data_dir = '../'
dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour to get behaviour.tsv
texts = 'Bili_2M_item.csv'

max_seq_len = 21
min_seq_len = 5
num_words_title = 30
which_language = 'zh'  # zh, en

freeze_paras_before_list = [0]
NLP_model_load_list = ['chinese_bert_wwm']  # bert_base_uncased, chinese_bert_wwm

mode = 'test'
item_tower_list = ['txt']
model_tower = 'DSSM'

load_ckpt_name_list = ['None']

dnn_layers_list = [0]
l2_weight_list = [0.1]
drop_rate = 0.1
batch_size_list = [1024]
lr_list = [1e-4]
embedding_dim = 1024
fine_tune_lr_list = [1e-5]


for i in range(len(load_ckpt_name_list)):
    item_tower = item_tower_list[i]
    lr = lr_list[i]
    load_ckpt_name = load_ckpt_name_list[i]
    NLP_model_load = NLP_model_load_list[i]
    freeze_paras_before = freeze_paras_before_list[i]
    batch_size = batch_size_list[i]
    fine_tune_lr = fine_tune_lr_list[i]
    dnn_layers = dnn_layers_list[i]
    l2_weight = l2_weight_list[i]
    label_screen = '{}_modality-{}_bs-{}_ed-{}_lr-{}_dp-{}_dnnL-{}_L2-{}_Flr-{}_ckp-{}'.format(
        item_tower, NLP_model_load, batch_size, embedding_dim, lr,
        drop_rate, dnn_layers, l2_weight, fine_tune_lr, load_ckpt_name)
    run_test_py = "CUDA_VISIBLE_DEVICES='0' \
              python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
              run_test.py --root_data_dir {}  --dataset {} --behaviors {} --texts {}\
              --mode {} --item_tower {} --load_ckpt_name {} --label_screen {}\
              --dnn_layers {} --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
              --NLP_model_load {} --freeze_paras_before {} --fine_tune_lr {}\
              --which_language {} --num_words_title {} --max_seq_len {} --min_seq_len {} \
              --model_tower {} --gpu_device {} \
              ".format(
                root_data_dir, dataset, behaviors, texts,
                mode, item_tower, load_ckpt_name, label_screen,
                dnn_layers, l2_weight, drop_rate, batch_size, lr, embedding_dim,
                NLP_model_load, freeze_paras_before, fine_tune_lr,
                which_language, num_words_title, max_seq_len, min_seq_len,
                model_tower, gpu_device)
    os.system(run_test_py)
