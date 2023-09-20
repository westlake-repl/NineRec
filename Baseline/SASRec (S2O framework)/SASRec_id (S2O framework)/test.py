import os

gpu_device = '0-7_135'

root_data_dir = '../'
dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour.py to get behaviour.tsv
texts = 'Bili_2M_item.csv'

max_seq_len = 20
min_seq_len = 5
num_words_title = 30
which_language = 'zh'

freeze_paras_before_list = [0]
NLP_model_load_list = ['chinese_bert_wwm']  # bert_base_uncased, chinese_bert_wwm

mode = 'test'
item_tower_list = ['id']
model_tower = 'sasrec'

model_path = './XXX/XXX/'  # checkpoint path
load_ckpt_name_list = ['None']  # checkpoint file

block_num_list = [2]
l2_weight_list_R = [0.1]
l2_weight_list_M = [0.1]
drop_rate = 0.1
batch_size_list = [8]
lr_list = [1e-5]
embedding_dim = 1024
fine_tune_lr_list = [1e-5]


for i in range(len(load_ckpt_name_list)):
    block_num = block_num_list[i]
    item_tower = item_tower_list[i]
    lr = lr_list[i]
    load_ckpt_name = load_ckpt_name_list[i]
    NLP_model_load = NLP_model_load_list[i]
    freeze_paras_before = freeze_paras_before_list[i]
    batch_size = batch_size_list[i]
    fine_tune_lr = fine_tune_lr_list[i]
    l2_weight_R = l2_weight_list_R[i]
    l2_weight_M = l2_weight_list_M[i]
    label_screen = '{}_modality-{}_bs-{}_ed-{}_bn-{}_lr-{}_Flr-{}_L2r{}_L2m{}_dp-{}_ckp-{}'.format(
        item_tower, NLP_model_load, batch_size, embedding_dim, block_num, lr, fine_tune_lr,
        l2_weight_R, l2_weight_M, drop_rate, load_ckpt_name)
                                        
    run_test_py = "CUDA_VISIBLE_DEVICES='0' \
              /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
              run_test.py --root_data_dir {}  --dataset {} --behaviors {} --texts {}\
              --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --block_num {}\
              --l2_weight_R {} --l2_weight_M {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
              --NLP_model_load {} --freeze_paras_before {} --fine_tune_lr {}\
              --which_language {} --num_words_title {} --max_seq_len {} --min_seq_len {} \
              --model_path {} --model_tower {} --gpu_device {} \
              ".format(
                root_data_dir, dataset, behaviors, texts,
                mode, item_tower, load_ckpt_name, label_screen, block_num,
                l2_weight_R, l2_weight_M, drop_rate, batch_size, lr, embedding_dim,
                NLP_model_load, freeze_paras_before, fine_tune_lr,
                which_language, num_words_title, max_seq_len, min_seq_len,
                model_path, model_tower, gpu_device)
    os.system(run_test_py)
