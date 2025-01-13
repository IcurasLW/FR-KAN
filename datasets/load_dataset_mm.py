"""Implements dataloaders for the AVMNIST dataset.

Here, the data is assumed to be in a folder titled "avmnist".
"""
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import h5py

import pickle
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import torch



def get_avmnist(data_dir, batch_size=128, num_workers=8, train_shuffle=True, flatten_audio=False, flatten_image=False, unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):
    """Get dataloaders for AVMNIST.

    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        flatten_audio (bool, optional): Whether to flatten audio data or not. Defaults to False.
        flatten_image (bool, optional): Whether to flatten image data or not. Defaults to False.
        unsqueeze_channel (bool, optional): Whether to unsqueeze any channels or not. Defaults to True.
        generate_sample (bool, optional): Whether to generate a sample and save it to file or not. Defaults to False.
        normalize_image (bool, optional): Whether to normalize the images before returning. Defaults to True.
        normalize_audio (bool, optional): Whether to normalize the audio before returning. Defaults to True.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader, test dataloader)
    """
    
    # Some helper function
    def _saveimg(outa):
        from PIL import Image
        t = np.zeros((300, 300))
        for i in range(0, 100):
            for j in range(0, 784):
                imrow = i // 10
                imcol = i % 10
                pixrow = j // 28
                pixcol = j % 28
                t[imrow*30+pixrow][imcol*30+pixcol] = outa[i][j]
        newimage = Image.new('L', (300, 300))  # type, size
        
        newimage.putdata(t.reshape((90000,)))
        newimage.save("samples.png")


    def _saveaudio(outa):
        from PIL import Image
        t = np.zeros((340, 340))
        for i in range(0, 9):
            for j in range(0, 112*112):
                imrow = i // 3
                imcol = i % 3
                pixrow = j // 112
                pixcol = j % 112
                t[imrow*114+pixrow][imcol*114+pixcol] = outa[i][j]
        newimage = Image.new('L', (340, 340))  # type, size
        
        newimage.putdata(t.reshape((340*340,)))
        newimage.save("samples2.png")




    trains = [np.load(data_dir+"/image/train_data.npy"), 
              np.load(data_dir + "/audio/train_data.npy"), 
              np.load(data_dir+"/train_labels.npy")]
                                                        
    tests = [np.load(data_dir+"/image/test_data.npy"), 
             np.load(data_dir + "/audio/test_data.npy"), 
             np.load(data_dir+"/test_labels.npy")]

    transform = transforms.Resize((28))
    trains[1] = np.array(transform(torch.tensor(trains[1])))
    tests[1] = np.array(transform(torch.tensor(tests[1])))

    if flatten_audio:
        trains[1] = trains[1].reshape(60000, 112*112)
        tests[1] = tests[1].reshape(10000, 112*112)
    if generate_sample:
        _saveimg(trains[0][0:100])
        _saveaudio(trains[1][0:9].reshape(9, 112*112))
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if not flatten_image:
        trains[0] = trains[0].reshape(60000, 28, 28)
        tests[0] = tests[0].reshape(10000, 28, 28)
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    
    trainlist = [[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist = [[tests[j][i] for j in range(3)] for i in range(10000)]
    # valids = DataLoader(trainlist[55000:60000], shuffle=False,
    #                     num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(trainlist, shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size)
    
    input_shape = [28, 28, 1]
    n_classes = 10
    num_mod = 2
    return trains, tests, input_shape, n_classes, num_mod




def get_mimic(args):
    class MIMIC_Dataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            irg_ts = self.data[idx]['irg_ts']
            text = self.data[idx]['text_data']
            label = self.data[idx]['label']
            return label, irg_ts, text



    def text_pipeline(text):
        return [vocab[token] for token in tokenizer(text)]



    def collate_batch(batch, txt_len=3000, ts_len=64):
        label_list, text_list, ts_list = [], [], []
        
        for label, irg_ts, text in batch:
            text = " ".join(text)
            label_list.append(label)
            processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
            
            
            if len(processed_text) > txt_len:
                processed_text = processed_text[:txt_len]  # Truncate to max length
            else:
                padding = torch.full((txt_len - len(processed_text),), vocab["<pad>"], dtype=torch.int64)
                processed_text = torch.cat([processed_text, padding], dim=0)  # Pad to max length
            
            
            if len(irg_ts) > ts_len:
                irg_ts = torch.tensor(irg_ts[:ts_len, :])
            else:
                n_vars = irg_ts.shape[1]
                padding = torch.full((ts_len - len(irg_ts),), 0, dtype=torch.int64).unsqueeze(-1)
                padding = padding.repeat(1, n_vars)
                irg_ts = torch.cat([torch.tensor(irg_ts), padding], dim=0)
            
            
            text_list.append(processed_text)
            ts_list.append(irg_ts)
            
        labels = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.stack(text_list, dim=0)
        ts_list = torch.stack(ts_list, dim=0)
        return text_list, ts_list, labels


    tokenizer = get_tokenizer("basic_english")
    data_dir = args.data_path
    with open(os.path.join(data_dir, 'MIMIC-III', 'trainp2x_data.pkl'), 'rb') as train_f:
        train_data = pickle.load(train_f)
        
    with open(os.path.join(data_dir, 'MIMIC-III', 'testp2x_data.pkl'), 'rb') as test_f:
        test_data = pickle.load(test_f)

    def yield_tokens(data_iter):
        for each in data_iter:
            if 'text_data' in each.keys():
                txt = each['text_data']
                yield tokenizer(" ".join(txt))
            else:
                yield ''


    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    num_tokens = len(vocab)
    
    train_dataset = []
    test_dataset = []
    ## Remove those missing text or missing ts
    for idx, each in enumerate(train_data):
        if'text_data' in each and 'text_data' in each:
            train_dataset.append(each)


    for idx, each in enumerate(test_data):
        if 'irg_ts' in each and 'text_data' in each:
            test_dataset.append(each)


    del train_data
    del test_data


    txt_len = 3000
    ts_len = 64
    n_class = 2
    n_vars_ts = 17
    
    train_dataset = MIMIC_Dataset(train_dataset)
    test_dataset = MIMIC_Dataset(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_batch)

    return train_loader, test_loader, num_tokens, txt_len, ts_len, n_vars_ts, n_class



def get_loader(args):
    if args.data_name == 'AVMNIST':
        return get_avmnist(data_dir=args.data_path + '/AVMNIST/avmnist')
    elif args.data_name == 'MIMIC':
        return get_mimic(args)