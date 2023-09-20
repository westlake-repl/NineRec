import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = '1'



if __name__ == '__main__':

    device = '0,1,2,3,4,5,6,7'

    import random
    master_port = random.randint(1002,9999)
    
    nproc_per_node = len(device.split(','))
        
    run_yaml = f"CUDA_VISIBLE_DEVICES='{device}' /opt/anaconda3/envs/myenv/bin/python  -m torch.distributed.run --nproc_per_node {nproc_per_node} \
    --master_port {master_port} run.py --config_file YAML/dssm_bert.yaml"

    os.system(run_yaml)

 

