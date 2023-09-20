import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(os.path.join(BASE_DIR,".."))



def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
    parser.add_argument("--item_tower", type=str, default="modal", choices=['modal',"ID", "text-only", "CV-only"])
    parser.add_argument("--root_data_dir", type=str, default=root_data_dir)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--behaviors", type=str, default=None)
    parser.add_argument("--texts", type=str, default=None)
    parser.add_argument("--lmdb_data", type=str, default=None) 

    # ============== train parameters==============
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--drop_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    # ============== model parameters for text ==============
    parser.add_argument("--bert_model_load", type=str, default=None)
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--text_freeze_paras_before", type=int, default=None)
    parser.add_argument("--text_fine_tune_lr", type=float, default=None)
    parser.add_argument("--num_words_title", type=int, default=50)

    # ============== model parameters for image ==============
    parser.add_argument("--CV_model_load", type=str, default=None)
    parser.add_argument("--CV_freeze_paras_before", type=int, default=None)
    parser.add_argument("--CV_resize", type=int, default=None)
    parser.add_argument("--CV_fine_tune_lr", type=float, default=None)

    # ============== model parameters ==============
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)

    # ============== SasRec的结构参数 ==============
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)

    # ==============  GruRec and NextItnet ==============
    parser.add_argument("--block_num", type=int, default=2)

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default=None)
    parser.add_argument("--label_screen", type=str, default=None)
    parser.add_argument("--logging_num", type=int, default=None)
    parser.add_argument("--testing_num", type=int, default=None)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--node_rank", default=0, type=int)

    # ============== fushion methods ==============
    parser.add_argument("--fusion_method", type=str, default="merge_attn", choices=['co_att', 'merge_attn','sum', 'concat', 'film', 'gated'])
    parser.add_argument("--num_co_attn_fuse_layers", type=int, default=1)
    parser.add_argument("--num_merge_attn_fuse_layers", type=int, default=1)

    # ==============various rec methods ==============
    parser.add_argument("--benchmark", type=str, default=None)

    parser.add_argument("--scheduler", type=str, default="step_schedule", choices=['cosine_schedule', 'linear_schedule', "step_schedule"])

    parser.add_argument("--scheduler_steps", type=int, default=100)

    args = parser.parse_args()

    return args

