from torch.utils.data import Dataset
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

from utils import new_utils


class Text2MotionDataset(Dataset):
    def __init__(self, 
                 dataname,
                 is_eval, 
                 usage='train',
                 max_text_len=20, 
                 unit_length=4):
        self.dataname = dataname
        self.is_eval = is_eval
        self.usage = usage
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        
        self.w_vectorizer = new_utils.load_word_vectorizer()

        dataset_info = new_utils.DatasetInfo(dataname, usage=usage)
        self.data_root = dataset_info.data_root
        self.motion_dir = dataset_info.motion_dir
        self.text_dir = dataset_info.text_dir
        self.joints_num = dataset_info.joints_num
        self.max_motion_length = dataset_info.max_motion_length
        fps = dataset_info.fps
        min_motion_length = dataset_info.min_motion_length
        self.mean = dataset_info.mean
        self.std = dataset_info.std
        id_list = dataset_info.id_list

        #* action embedding
        llm_version = 'qwen3-8b'
        self.max_action_num = 5
        if dataname == 't2m':
            self.llm_text_dir = pjoin(new_utils.data_root, 'LLM_processed', 'HumanML3D', llm_version)
        else:
            self.llm_text_dir = pjoin(new_utils.data_root, 'LLM_processed', 'KIT-ML', llm_version)
        print('LLM version: ', llm_version)

        data_dict = {}
        new_name_list = []
        length_list = []
        
        for name in tqdm(id_list):
            
            #* action embedding
            llm_text_path = pjoin(self.llm_text_dir, name + '.txt')
            with open(llm_text_path, "r") as f:
                llm_text = f.readlines()

            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_length or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False

                line_i = 0
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens

                        #* action embedding
                        llm_caption = llm_text[line_i].rstrip('\n')
                        text_dict['llm_caption'] = llm_caption

                        action_num = len(llm_caption.split('#'))
                        action_num = max(1, min(action_num, self.max_action_num))
                        
                        line_i += 1

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps):int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_length or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                
                # whole motion sequence
                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            
            except Exception as e:
                pass

        if self.is_eval:
            # order samples according to motion length if evaling
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list = new_name_list

        self.data_dict = data_dict
        self.name_list = name_list
        self.length_arr = np.array(length_list)

        #* evaluation
        self.pointer = 0
        if self.is_eval:
            self.max_length = 20
            self.reset_max_len(self.max_length)

        print("Total number of motions for {}ing Transformer: {}".format(usage, len(self.data_dict)))
       
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        
        #* action embedding
        llm_caption = text_data['llm_caption']
        llm_captions = llm_caption.split('#')[:self.max_action_num]

        #* evaluation
        if self.is_eval:
            if len(tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)

            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        #* motion length
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        motion = (motion - self.mean) / self.std

        #* motion segmentation
        action_num = len(llm_captions)
        t_length = m_length // self.unit_length
        base_len = t_length // action_num
        remainder = t_length % action_num
        action_ids = np.arange(action_num)
        motion_seg = base_len + (action_ids < remainder).astype(int)

        motion_seg = (motion_seg.tolist() + [0] * self.max_action_num)[:self.max_action_num]

        #* motion length
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))], 
                                     axis=0)

        if self.is_eval:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), llm_captions, motion_seg
        else:
            return caption, motion, m_length, llm_captions, motion_seg
