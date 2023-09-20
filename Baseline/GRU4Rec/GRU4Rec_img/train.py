import os

gpu_device = '0-7_135'

root_data_dir = '../'
dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour.py to get behaviour.tsv
images = 'Bili_2M_item.csv'
lmdb_data = 'Bili_2M.lmdb'  # run get_lmdb.py to get lmdb database

logging_num = 4
testing_num = 1

max_seq_len = 20
min_seq_len = 5

CV_resize = 224
CV_model_load = 'resnet50'  # 'resnet50', 'swin_tiny', 'swin_small', 'swin_base'
freeze_paras_before = 0

mode = 'train'
item_tower = 'img'
model_tower = 'gru4rec'
is_pretrain = 0  # 0, 1

epoch = 120
model_path = './XXX/XXX/'  # checkpoint path
load_ckpt_name = 'None'  # checkpoint file
num_workers = 12

block_num_list = [2]
l2_weight_list_R = [0.1]
l2_weight_list_M = [0.0]
drop_rate_list = [0.1]
batch_size_list = [8]
lr_list = [1e-4]
embedding_dim_list = [1024]
fine_tune_lr_list = [1e-5]

for block_num in block_num_list:
    for l2_weight_R in l2_weight_list_R:
        for l2_weight_M in l2_weight_list_M:
            for batch_size in batch_size_list:
                for drop_rate in drop_rate_list:
                    for embedding_dim in embedding_dim_list:
                        for fine_tune_lr in fine_tune_lr_list:
                            for lr in lr_list:
                                label_screen = '{}_bs{}_ed{}_bn{}_lr{}_Flr{}_dp{}_L2r{}_L2m{}'.format(
                                    item_tower, batch_size, embedding_dim, block_num,
                                    lr, fine_tune_lr, drop_rate, l2_weight_R, l2_weight_M)
                                run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
                                         python  -m torch.distributed.launch --nproc_per_node 8 --master_port 1234\
                                         run.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
                                         --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                         --l2_weight_R {} --l2_weight_M {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} --block_num {}\
                                         --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}\
                                         --model_tower {} --max_seq_len {} --min_seq_len {} --num_workers {}\
                                         --model_path {} --is_pretrain {} --gpu_device {}\
                                         ".format(
                                            root_data_dir, dataset, behaviors, images, lmdb_data,
                                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                            l2_weight_R, l2_weight_M, drop_rate, batch_size, lr, embedding_dim, block_num,
                                            CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr,
                                            model_tower, max_seq_len, min_seq_len, num_workers,
                                            model_path, is_pretrain, gpu_device)
                                os.system(run_py)
