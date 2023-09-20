import os

gpu_device = '0_132'

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
item_tower = 'id'
model_tower = 'DSSM'

epoch = 120
load_ckpt_name = 'None'
num_workers = 12

dnn_layers_list = [0]
l2_weight_list = [0.1]
dropout_list = [0.1]
batch_size_list = [4096]
embedding_dim_list = [1024]
lr_list = [1e-4, 5e-5, 1e-5]

for l2_weight in l2_weight_list:
    for drop_rate in dropout_list:
        for dnn_layers in dnn_layers_list:
            for embedding_dim in embedding_dim_list:
                for batch_size in batch_size_list:
                    for lr in lr_list:
                        fine_tune_lr = 0
                        label_screen = '{}_bs{}_ed{}_dnnL{}_lr{}_Flr{}_dp{}_L2{}'.format(
                            item_tower, batch_size, embedding_dim, dnn_layers,
                            lr, fine_tune_lr, drop_rate, l2_weight)
                        run_py = "CUDA_VISIBLE_DEVICES='0,1' \
                                     python  -m torch.distributed.launch --nproc_per_node 2 --master_port 1234\
                                     run.py --root_data_dir {}  --dataset {} --behaviors {} --texts {}\
                                     --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                     --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} --dnn_layers {} \
                                     --NLP_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {} \
                                     --gpu_device {} --num_words_title {} --which_language {} --max_seq_len {} --min_seq_len {} \
                                     --NLP_model_load {} --model_tower {} --num_workers {}".format(
                                        root_data_dir, dataset, behaviors, texts,
                                        mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                        l2_weight, drop_rate, batch_size, lr, embedding_dim, dnn_layers,
                                        NLP_model_load, epoch, freeze_paras_before, fine_tune_lr,
                                        gpu_device, num_words_title, which_language, max_seq_len, min_seq_len,
                                        NLP_model_load, model_tower, num_workers)
                        os.system(run_py)
