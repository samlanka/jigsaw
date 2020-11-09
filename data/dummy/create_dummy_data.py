import os
import sys
sys.path.append(os.path.abspath('..'))

from tqdm import tqdm
from utils import set_device
from utils import gpt2_tokenizer
from utils import torch2scipy_sparse
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
import dill as pickle
import numpy as np
from copy import deepcopy


def create_bow_matrix(corpus_ids_tensor, split_ids, file_path=None):
    """
    Creates and saves sparse matrix of token counts of size num_docs x vocab_size
    :param corpus_ids_tensor: (torch.LongTensor) token indices of size num_docs x seq_len
    :param split_ids: (list) indices for splitting into independent documents
    :param file_path: (str) path to store document-term counts sparse matrix
    """
    corpus_bow = []
    _, seq_len = corpus_ids_tensor.size()
    split_sizes = np.diff(split_ids).tolist()
    num_docs = len(split_sizes)
    vocab_size = len(gpt2_tokenizer)
    unk_id = gpt2_tokenizer.unk_token_id

    doc_ids_list = corpus_ids_tensor.split(split_sizes, dim=0)
    print('Creating doc-term counts matrix')
    for doc_ids in tqdm(doc_ids_list, total=len(split_sizes)):
        token_ids = doc_ids.flatten().tolist()
        num_tokens = len(token_ids)
        row_id = [0] * num_tokens
        col_id = list(range(num_tokens))
        ids = torch.LongTensor([row_id, col_id, token_ids])
        token_count = torch.ones(num_tokens)
        doc_bow = torch.sparse.LongTensor(ids, token_count, torch.Size([1, num_tokens, vocab_size]))
        doc_bow = torch.sparse.sum(doc_bow, dim=1)
        corpus_bow.append(doc_bow)

    corpus_bow = torch.cat(corpus_bow, dim=0)
    assert list(corpus_bow.shape) == [num_docs, vocab_size]
    corpus_bow = torch2scipy_sparse(corpus_bow)
    corpus_bow[:, unk_id] = 0  # setting count of unk_token to 0

    path = './dummy/DocTerm_counts_sparse_gpt2Tok.pkl' if not file_path else file_path
    with open(path, 'wb') as f:
        print(f'Saving file to {os.path.abspath(path)}')
        pickle.dump(corpus_bow, f)


def get_gpt2_prob(ip_tensor, op_tensor, batch_size, device='cpu', save=True, file_path=None):
    """
    Returns GPT-2 prediction probabilities of each target token in a document.
    :param ip_tensor: (torch.LongTensor) input token indices of size num_examples x seq_len
    :param op_tensor: (torch.LongTensor) output token indices of size num_examples x seq_len
    :param batch_size: (int) batch size
    :param device: (str) 'cpu'/'cuda'
    :param save: (bool) flag to store GPT-2 predictions probabilities
    :param file_path: (str) path to store GPT-2 prediction probabilities
    :return: GPT-2 token prediction probabilities of size num_docs x seq_len
    :rtype: torch.FloatTensor
    """
    softmax = nn.Softmax(dim=-1)
    gpt2 = GPT2LMHeadModel.from_pretrained('distilgpt2', cache_dir='../models/')
    gpt2.to(device)
    gpt2.eval()

    num_examples, seq_len = ip_tensor.size()
    ip_batches = ip_tensor.split(batch_size, dim=0)
    op_batches = op_tensor.split(batch_size, dim=0)
    gpt2_probs = []

    print('Calculating GPT-2 prediction probabilities')
    for ip, op in tqdm(zip(ip_batches, op_batches), total=len(ip_batches)):
        ip = ip.to(device)
        op = op.unsqueeze(-1).to(device)
        with torch.no_grad():
            log_probs, _ = gpt2(ip)
        probs = softmax(log_probs)
        probs_target_token = probs.gather(-1, op)  # batch, seq_len, 1
        gpt2_probs.append(probs_target_token.squeeze(-1))

    gpt2_probs = torch.cat(gpt2_probs, dim=0)
    assert list(gpt2_probs.shape) == [num_examples, seq_len]

    if save:
        path = './dummy/DocTerm_gpt2_probs.pt' if not file_path else file_path
        print(f'Saving file to {os.path.abspath(path)}')
        torch.save(gpt2_probs.cpu(), path)
    return gpt2_probs.cpu()


def encode_corpus(corpus, seq_len, min_token_len=1, max_sections_per_email=3, save_new_corpus=True,
                  file_path=None):
    """
    Tokenizes, filters and converts text corpus into sequences of token indices. Generates input and output tensors.
    :param corpus: (list(str)) list of documents
    :param seq_len: (int) length of each sequence
    :param min_token_len: (int) minimum number of characters in a token
    :param max_sections_per_email: (int) maximum number of chunks of seq_len for each email
    :param save_new_corpus: (bool) flag to save the filtered text corpus
    :param file_path: (str) path to save the filtered text corpus
    :return: input tensor, output tensor, list of split indices for each document
    :rtype: (torch.LongTensor, torch.LongTensor, list)
    """
    corpus_final = deepcopy(corpus)
    c = 0  # tracks the index in corpus_final

    split_ids = [0]
    unk_id = gpt2_tokenizer.unk_token_id
    ip_tensor, op_tensor = [], []

    print('Encoding corpus')
    for i, example_i in tqdm(enumerate(corpus), total=len(corpus)):
        tokens = gpt2_tokenizer.tokenize(example_i)
        tokens = list(filter(lambda tok: len(tok) >= min_token_len, tokens))
        tokens = gpt2_tokenizer.convert_tokens_to_ids(tokens)
        ip_ids = tokens[:-1]
        op_ids = tokens[1:]
        num_tokens = len(op_ids)
        if num_tokens < 1:
            del corpus_final[c]
            continue
        remainder = num_tokens % seq_len
        num_sections = int(num_tokens // seq_len) + int(remainder != 0)
        num_sections = min(num_sections, max_sections_per_email)
        corpus_final[c] = gpt2_tokenizer.decode(tokens[:(num_sections * seq_len) + 1])
        c += 1
        split_ids.append(split_ids[-1])

        for j in range(num_sections):
            ip_ids_j = ip_ids[j * seq_len:(j + 1) * seq_len]
            ip_ids_j = ip_ids_j + [unk_id] * (seq_len - len(ip_ids_j))
            ip_tensor.append(ip_ids_j)

            op_ids_j = op_ids[j * seq_len:(j + 1) * seq_len]
            op_ids_j = op_ids_j + [unk_id] * (seq_len - len(op_ids_j))
            op_tensor.append(op_ids_j)
            # assert len(ip_ids_j) == seq_len
            # assert len(op_ids_j) == seq_len
            split_ids[-1] += 1

    ip_tensor = torch.LongTensor(ip_tensor)
    op_tensor = torch.LongTensor(op_tensor)
    if save_new_corpus:
        path = './dummy/corpus_final.pkl' if not file_path else file_path
        print(f'Saving filtered corpus to {os.path.abspath(path)}')
        with open(path, 'wb') as f:
            pickle.dump(corpus_final, f)

    return ip_tensor, op_tensor, split_ids


def main():
    str1 = 'An apple a day keeps the doctor away!'
    str2 = 'The Abel Prize is sometimes called "the Nobel prize of mathematics".'
    str3 = 'Under the Paris Agreement, each country must determine, plan, and regularly report on the contribution ' \
           'that it undertakes to mitigate global warming.'

    corpus = [str1, str2, str3]

    device = set_device()
    seq_len = 1024
    max_sections_per_example = 3  # clip each example to have <= max_sections_per_example * seq_len tokens
    min_token_len = 1  # filter tokens of length < min_token_len
    batch_size = 8

    ip_tensor, op_tensor, split_ids = encode_corpus(corpus, seq_len, min_token_len, max_sections_per_example)
    torch.save(ip_tensor, './dummy/ip_tensor.pt')
    torch.save(op_tensor, './dummy/op_tensor.pt')
    with open('./dummy/split_ids.pkl', 'wb') as f:
        pickle.dump(split_ids, f)

    create_counts_bow(op_tensor, split_ids)
    gpt2_probs = get_gpt2_prob(ip_tensor, op_tensor, batch_size, device, save=True)


if __name__ == '__main__':
    main()
