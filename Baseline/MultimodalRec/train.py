import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(os.path.join(BASE_DIR,".."))


dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour to get behaviour.tsv
texts = 'Bili_2M_item.csv'
lmdb_data = 'Bili_2M.lmdb'  # run get_lmdb.py to get lmdb database

logging_num = 10
testing_num = 1

CV_resize = 224

CV_model_load = "swin-tiny-patch4-window7-224"
# CV_model_load = "vit-mae-base"
# CV_model_load = "swin-base-patch4-window7-224"
# CV_model_load = "clip-vit-base-patch32"

BERT_model_load =  "chinese_bert_wwm" 
# BERT_model_load =  "chinese-roberta-wwm-ext" 
# BERT_model_load =  "roberta-base" 
# BERT_model_load =  "xlm-roberta-base"

CV_freeze_paras_before = 0
text_freeze_paras_before = 0

CV_fine_tune_lr = 1e-4
text_fine_tune_lr = 1e-4


mode = 'train' # train test
item_tower = 'modal'

epoch = 150
load_ckpt_name = 'None'

l2_weight_list = [0.1]
drop_rate_list = [0.1]
batch_size_list = [128]
lr_list = [1e-4]
embedding_dim_list = [768]
max_seq_len_list = [20]


benchmark_list = ['sasrec']
fusion_method = "gated"  # ['co_att', 'merge_attn','sum', 'concat', 'film', 'gated']

scheduler_steps = 120

for weight_decay in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for embedding_dim in embedding_dim_list:
                for lr in lr_list:
                    for max_seq_len in max_seq_len_list:
                            for benchmark in benchmark_list:
                                label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                                        item_tower, batch_size, embedding_dim, lr,
                                        drop_rate, weight_decay, max_seq_len)

                                run_py = "CUDA_VISIBLE_DEVICES='4,5,6,7' \
                                        python  -m torch.distributed.launch --nproc_per_node 4 --master_port 1264  run.py\
                                        --root_data_dir {}  --dataset {} --behaviors {} --texts {}  --lmdb_data {}\
                                        --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                        --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                        --CV_resize {} --CV_model_load {} --bert_model_load {}  --epoch {} \
                                        --text_freeze_paras_before {} --CV_freeze_paras_before {} --max_seq_len {} \
                                        --CV_fine_tune_lr {} --text_fine_tune_lr {} --fusion_method {} --benchmark {} --scheduler_steps {}".format(
                                    root_data_dir, dataset, behaviors, texts, lmdb_data,
                                    mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                    weight_decay, drop_rate, batch_size, lr, embedding_dim,
                                    CV_resize, CV_model_load, BERT_model_load, epoch,
                                    text_freeze_paras_before, CV_freeze_paras_before, max_seq_len,
                                    CV_fine_tune_lr, text_fine_tune_lr, fusion_method, benchmark, scheduler_steps)

                                os.system(run_py)




