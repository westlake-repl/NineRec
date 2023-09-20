from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
    parser.add_argument("--item_tower", type=str, default="img", choices=['img', 'id'])
    parser.add_argument("--model_tower", type=str, default="sasrec")
    parser.add_argument("--root_data_dir", type=str, default="../",)
    parser.add_argument("--dataset", type=str, default='Bilibili')
    parser.add_argument("--behaviors", type=str, default='behaviors.tsv')
    parser.add_argument("--images", type=str, default='images_log.csv')
    parser.add_argument("--lmdb_data", type=str, default='image.lmdb')

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight_R", type=float, default=0)
    parser.add_argument("--l2_weight_M", type=float, default=0)
    parser.add_argument("--drop_rate", type=float, default=0.1)

    # ============== model parameters ==============
    parser.add_argument("--CV_model_load", type=str, default='resnet-50',
                        choices=['resnet50', 'swin_tiny', 'swin_base', 'swin_small', 'swin_base'])
    parser.add_argument("--freeze_paras_before", type=int, default=45)
    parser.add_argument("--CV_resize", type=int, default=224)
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--min_seq_len", type=int, default=3)

    # ============== Nextitnet ==============
    parser.add_argument('--dilations', type=int, default=[1, 4], help='Number of transformer layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='Number of heads for multi-attention')
    parser.add_argument('--block_num', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--pad_token', type=int, default=0)

    # ============== transfer learning ==============
    parser.add_argument('--is_pretrain', type=int, default=1, help='0: pretrain mode, 1: transfer mode')

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--model_path", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)
    
    # ============== others ==============
    parser.add_argument("--gpu_device", type=str, default='5,6,7,8_225')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
