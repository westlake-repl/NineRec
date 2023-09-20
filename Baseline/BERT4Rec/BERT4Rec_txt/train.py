import os

gpu_device = '0,1,2,3_225'

root_data_dir = '../'
dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour to get behaviour.tsv
texts = 'Bili_2M_item.csv'

logging_num = 4
testing_num = 1

max_seq_len = 21
min_seq_len = 5
num_words_title = 30
which_language = 'zh'  # zh, en

NLP_model_load = 'chinese_bert_wwm'  # bert_base_uncased, chinese_bert_wwm
freeze_paras_before = 0

mode = 'train'
item_tower = 'txt'
model_tower = 'bert4rec'
is_pretrain = 0  # 0, 1

epoch = 120
model_path = './XXX/XXX/'  # checkpoint path
load_ckpt_name = 'None'  # checkpoint file
num_workers = 12

mask_prob_list = [0.4]
l2_weight_list = [0.1]
drop_rate_list = [0.1]
batch_size_list = [10]
lr_list = [1e-4]
embedding_dim_list = [256]
fine_tune_lr_list = [1e-5]

for mask_prob in mask_prob_list:
    for l2_weight in l2_weight_list:
        for batch_size in batch_size_list:
            for drop_rate in drop_rate_list:
                for embedding_dim in embedding_dim_list:
                    for fine_tune_lr in fine_tune_lr_list:
                        for lr in lr_list:
                            label_screen = '{}_bs{}_ed{}_mp{}_lr{}_Flr{}_dp{}_L2{}'.format(
                                item_tower, batch_size, embedding_dim, mask_prob,
                                lr, fine_tune_lr, drop_rate, l2_weight)
                            run_py = "CUDA_VISIBLE_DEVICES='9' \
                                     python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1231\
                                     run.py --root_data_dir {}  --dataset {} --behaviors {} --texts {}\
                                     --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                     --mask_prob {} --l2_weight {} --drop_rate {} --batch_size {} --lr {}\
                                     --embedding_dim {} --epoch {} --freeze_paras_before {}  --fine_tune_lr {}\
                                     --model_tower {} --max_seq_len {} --min_seq_len {} --num_words_title {} --num_workers {}\
                                     --which_language {} --model_path {} --NLP_model_load {} --is_pretrain {} --gpu_device {}\
                                     ".format(
                                        root_data_dir, dataset, behaviors, texts,
                                        mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                        mask_prob, l2_weight, drop_rate, batch_size, lr,
                                        embedding_dim, epoch, freeze_paras_before, fine_tune_lr,
                                        model_tower, max_seq_len, min_seq_len, num_words_title, num_workers,
                                        which_language, model_path, NLP_model_load, is_pretrain, gpu_device)
                            os.system(run_py)
