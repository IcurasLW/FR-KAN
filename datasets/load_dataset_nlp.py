
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
# Define tokenizer and device


def get_loader(args):
    
    # Computer Vision Dataset
    if args.data_name == 'AG_NEWS':

        tokenizer = get_tokenizer("basic_english")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the AG_NEWS dataset
        train_iter, test_iter = AG_NEWS(split=('train', 'test'))

        # Function to yield tokens from the dataset
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        # Build vocabulary from training data
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])

        # Define pipelines for text and labels
        def text_pipeline(text):
            return [vocab[token] for token in tokenizer(text)]

        def label_pipeline(label):
            return label - 1  # Convert labels to 0-based index (1 -> 0, 2 -> 1, etc.)



        # Define collate function for DataLoader
        # def collate_batch(batch):
        #     label_list, text_list = [], []
        #     for label, text in batch:
        #         label_list.append(label_pipeline(label))
        #         processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        #         text_list.append(processed_text)
        #     labels = torch.tensor(label_list, dtype=torch.int64)
        #     texts = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
        #     return texts.to(device), labels.to(device)
        
        
        
        def collate_batch(batch, max_length=128):
            label_list, text_list = [], []
            # truncated_padded_texts = []
            if max_length is None:
                batch_max_length = max(len(text) for text in text_list)
            else:
                batch_max_length = max_length
                
            for label, text in batch:
                label_list.append(label_pipeline(label))
                processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
                if len(processed_text) > batch_max_length:
                    truncated_text = processed_text[:batch_max_length]  # Truncate to max length
                else:
                    padding = torch.full((batch_max_length - len(processed_text),), vocab["<pad>"], dtype=torch.int64)
                    truncated_text = torch.cat([processed_text, padding], dim=0)  # Pad to max length
                text_list.append(truncated_text)
            labels = torch.tensor(label_list, dtype=torch.int64)
            texts = torch.stack(text_list, dim=0)
            return texts.to(device), labels.to(device)


        train_iter, test_iter = AG_NEWS(split=('train', 'test'))  # Reload iterators
        train_dataloader = DataLoader(list(train_iter), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(list(test_iter), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)


        unique_tokens = set()
        n_classes = []
        for label, text in train_iter:
            unique_tokens.update(tokenizer(text))
            n_classes.append(label)
        n_classes = len(set(n_classes))
        num_tokens = len(unique_tokens) + 2 # Add two special tokens for padding
        embedding_dim = 128
        max_length = 128
        # ipnut_shape = num_tokens * embedding_dim
        return train_dataloader, test_dataloader, max_length, num_tokens, embedding_dim, n_classes

