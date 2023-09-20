import os

gpu_device = '0-7_135'

root_data_dir = '../'
dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour.py to get behaviour.tsv
images = 'Bili_2M_item.csv'
lmdb_data = 'Bili_2M.lmdb'  # run get_lmdb.py to get lmdb database

max_seq_len = 20
min_seq_len = 5

CV_resize = 224
freeze_paras_before_list = [0]
CV_model_load_list = ['swin_small']  # 'resnet50', 'swin_tiny', 'swin_base'

mode = 'test'
item_tower_list = ['img']
model_tower = 'sasrec'

model_path = './XXX/XXX/'  # checkpoint path
load_ckpt_name_list = ['None']  # checkpoint file

block_num_list = [2]
l2_weight_list_R = [0.1]
l2_weight_list_M = [0.0]
drop_rate = 0.1
batch_size_list = [16]
lr_list = [5e-5]
embedding_dim = 1024
fine_tune_lr_list = [5e-5]


for i in range(len(load_ckpt_name_list)):
    block_num = block_num_list[i]
    item_tower = item_tower_list[i]
    lr = lr_list[i]
    load_ckpt_name = load_ckpt_name_list[i]
    CV_model_load = CV_model_load_list[i]
    freeze_paras_before = freeze_paras_before_list[i]
    batch_size = batch_size_list[i]
    fine_tune_lr = fine_tune_lr_list[i]
    l2_weight_R = l2_weight_list_R[i]
    l2_weight_M = l2_weight_list_M[i]
    label_screen = '{}_modality-{}_bs-{}_ed-{}_bn-{}_lr-{}_Flr-{}_L2r-{}_L2m-{}_dp-{}_ckp-{}'.format(
        item_tower, CV_model_load, batch_size, embedding_dim, block_num,
        lr, fine_tune_lr, l2_weight_R, l2_weight_M, drop_rate, load_ckpt_name)
    run_test_py = "CUDA_VISIBLE_DEVICES='0,1,2,3' \
              python  -m torch.distributed.launch --nproc_per_node 4 --master_port 1235\
              run_test.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
              --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --block_num {}\
              --l2_weight_R {} --l2_weight_M {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
              --CV_resize {} --CV_model_load {} --freeze_paras_before {} --fine_tune_lr {}\
              --max_seq_len {} --min_seq_len {} \
              --model_path {} --model_tower {} --gpu_device {} \
              ".format(
                root_data_dir, dataset, behaviors, images, lmdb_data,
                mode, item_tower, load_ckpt_name, label_screen, block_num,
                l2_weight_R, l2_weight_M, drop_rate, batch_size, lr, embedding_dim,
                CV_resize, CV_model_load, freeze_paras_before, fine_tune_lr,
                max_seq_len, min_seq_len,
                model_path, model_tower, gpu_device)
    os.system(run_test_py)
