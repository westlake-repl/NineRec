
import torch
import numpy as np
import tqdm


def read_images(images_path):
    item_id_to_keys = {}
    item_name_to_id = {}
    index = 1
    with open(images_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            image_name = splited[0]
            item_name_to_id[image_name] = index
            item_id_to_keys[index] = u'{}'.format(int(image_name.replace('v', ''))).encode('ascii')
            # item_id_to_keys[index] = u'{}'.format(int(image_name)).encode('ascii')
            index += 1
    return item_name_to_id, item_id_to_keys

# read_behaviors_CV also works for ID-based
def read_behaviors_CV(behaviors_path, before_item_id_to_keys, before_item_name_to_id, max_seq_len, min_seq_len,Log_file):
    Log_file.info("##### images number {} {} (before clearing)#####".
                  format(len(before_item_id_to_keys), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))

    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_id = splited[0]
            history_item_name = str(splited[1]).strip().split(" ")
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_id] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))        
    Log_file.info("##### user seqs before {}".format(before_seq_num))

    item_id = 1
    item_id_to_keys = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_keys[item_id] = before_item_id_to_keys[before_item_id]
            item_id += 1
    
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {}, {}#####".format(item_num, item_id - 1, len(item_id_to_keys), len(item_id_before_to_now)))
    
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {} #用于计算score,需要LongTensor格式
    users_history_for_test = {} #用于计算score,需要LongTensor格式
    user_id = 0
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        users_train[user_id] = user_seq[:-2]
        users_valid[user_id] = user_seq[-(max_seq_len+2):-1]
        users_test[user_id] = user_seq[-(max_seq_len+1):]

        users_history_for_valid[user_id] = torch.LongTensor(np.array(users_train[user_id]))
        users_history_for_test[user_id] = torch.LongTensor(np.array(users_valid[user_id]))
        
        user_id += 1
        
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    return item_num, item_id_to_keys, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test


def read_texts(text_path, args, tokenizer):
    item_dic = {}
    item_name_to_index = {}
    item_index_to_name = {}
    index = 1
    with open(text_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            doc_name, title = splited
            item_name_to_index[doc_name] = index
            item_index_to_name[index] = doc_name
            index += 1
            # tokenizer
            title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)
            item_dic[doc_name] = [title]

    return item_dic, item_name_to_index, item_index_to_name

def read_behaviors_text(behaviors_path, item_dic, before_item_name_to_index, before_item_index_to_name, max_seq_len, min_seq_len, Log_file):
    Log_file.info("##### text number {} {} {} (before clearing)#####".
                  format(len(before_item_name_to_index), len(item_dic), len(before_item_index_to_name)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))

    before_item_num = len(before_item_name_to_index)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
        
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_id = splited[0]
            history_item_name = splited[1].split(" ")

            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_index[i] for i in history_item_name]
            user_seq_dic[user_id] = history_item_name
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1

    Log_file.info("##### pairs_num {}".format(pairs_num))
    Log_file.info("#### user seqs before {}".format(before_seq_num))

    for item_id in range(1, before_item_num + 1):
        if before_item_counts[item_id] == 0:
            item_dic.pop(before_item_index_to_name[item_id])

    item_id = 1
    item_num = len(item_dic)
    item_index = {}

    for doc_name, value in item_dic.items():
        item_index[doc_name] = item_id
        item_id += 1

    Log_file.info("##### items after clearing {}, {}#####".format(item_num, len(item_index)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    for user_name, user_seq_name in user_seq_dic.items():

        user_seq = [item_index[item_name] for item_name in user_seq_name]
        
        
        users_train[user_id] = user_seq[:-2]
        users_valid[user_id] = user_seq[-(max_seq_len+2):-1]
        users_test[user_id] = user_seq[-(max_seq_len+1):]

        users_history_for_valid[user_id] = torch.LongTensor(np.array(users_train[user_id])) 
        users_history_for_test[user_id] = torch.LongTensor(np.array(users_valid[user_id])) #用于计算score使用，提前生成，免得届时再生成

        user_id += 1

    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    return item_num, item_dic, item_index, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test


def get_doc_input_bert(news_dic, item_index, args):
    item_num = len(news_dic) + 1

    news_title = np.zeros((item_num, args.num_words_title), dtype='int32')
    news_title_attmask = np.zeros((item_num, args.num_words_title), dtype='int32')

    for key in news_dic:
        title = news_dic[key]
        doc_index = item_index[key]
        
        news_title[doc_index] = title[0]['input_ids'] #keys来自于tokenizer后形成的字典
        news_title_attmask[doc_index] = title[0]['attention_mask']

    return news_title, news_title_attmask
