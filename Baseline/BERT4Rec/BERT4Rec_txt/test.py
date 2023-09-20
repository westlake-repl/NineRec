import os

gpu_device = '0,1,2,3_225'

root_data_dir = '../'
dataset = 'Datasets/'
behaviors = 'Bili_2M_behaviour.tsv'  # run get_behaviour to get behaviour.tsv
texts = 'Bili_2M_item.csv'

max_seq_len = 21
min_seq_len = 5

mode = 'test'
item_tower_list = ['txt']
model_tower = 'bert4rec'

model_path = './XXX/XXX/'  # checkpoint path
load_ckpt_name_list = ['None']  # checkpoint file

mask_prob_list = [0.6]
l2_weight_list = [0.1]
drop_rate = 0.1
batch_size_list = [8]
lr_list = [1e-4]
embedding_dim = 256
fine_tune_lr_list = [1e-5]


for i in range(len(load_ckpt_name_list)):
    item_tower = item_tower_list[i]
    lr = lr_list[i]
    load_ckpt_name = load_ckpt_name_list[i]
    batch_size = batch_size_list[i]
    fine_tune_lr = fine_tune_lr_list[i]
    mask_prob = mask_prob_list[i]
    l2_weight = l2_weight_list[i]
    label_screen = '{}_modality_bs-{}_ed-{}_lr-{}_dp-{}_mp-{}_L2-{}_Flr-{}_ckp-{}'.format(
        item_tower, batch_size, embedding_dim, lr,
        drop_rate, mask_prob, l2_weight, fine_tune_lr, load_ckpt_name)
    run_test_py = "CUDA_VISIBLE_DEVICES='9' \
              python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
              run_test.py --root_data_dir {}  --dataset {} --behaviors {} --texts {}\
              --mode {} --item_tower {} --load_ckpt_name {} --label_screen {}\
              --mask_prob {} --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
              --fine_tune_lr {}\
              --max_seq_len {} --min_seq_len {} \
              --model_path {} --model_tower {} --gpu_device {} \
              ".format(
                root_data_dir, dataset, behaviors, texts,
                mode, item_tower, load_ckpt_name, label_screen,
                mask_prob, l2_weight, drop_rate, batch_size, lr, embedding_dim,
                fine_tune_lr,
                max_seq_len, min_seq_len,
                model_path, model_tower, gpu_device)
    os.system(run_test_py)
