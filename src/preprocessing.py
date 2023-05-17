import re
import argparse
import pandas as pd 
from nltk.stem.porter import PorterStemmer

# stem, stop words
stemmer = PorterStemmer()
def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')

    # Add data arguments
    parser.add_argument('--data', default='./data/', help='path to data directory')
    parser.add_argument('--index', default='./data/index', help='path to data directory')
    parser.add_argument('--dict_dir', default='./data/', help='path to dictionary')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    return args


def tokenize(words:str, using_stem:bool=True, del_stop:bool = False):
    # words.strip('\n').split(' '): the word may contain the punctuations : TODO
    # [\w+]: it may splite the url into several parts 
    for word in words.strip('\n').split(' '):
        if word == '...':
            a = 10
        if word == '':
            continue
        # 1. https
        # 2. split the  sentences by the punctuations 
        # 3. if it is started with hash marks, weights more 
        
        if re.search('http', word):
            continue
        
        for token in re.findall('#?\w+', word):
            if token == '...':
                a = 10
            if token == '':
                continue
            if re.match('^#', token):
                token = token[1:]
                if using_stem:
                    token = stemmer.stem(token)
                yield token, 2
            else:
                if using_stem:
                    token = stemmer.stem(token)
                yield token, 1


def main(args): 
    '''
    Parameter:
        output_dir: the direction to save the generated dictionary 
        sorce_files: the files that needed to be transformed into feature vectors 
    '''
    # generate the dictionary
    vocabulary = []
    
    freq = {}
    N_doc = 0
    N_vocab = 0
    with open(args.data+'train.csv', 'r') as f: 
        # generate the vocabulary basing on the training data 
        train_data = pd.read_csv(f)
        for idx in range(len(train_data)):
            tokens= train_data.text[idx]
            for word, _ in tokenize(tokens):
                if word in vocabulary:
                    freq[word] += 1
                else:
                    vocabulary.append(word)
                    freq[word] = 1
            N_doc += 1
        # the dictionary to convert the words into index 
        word2idx = {word:idx for idx, word in enumerate(vocabulary)}        
        doc_mat = [[0 for _ in range(len(vocabulary)+1)] for _ in range(N_doc)]
        # generate the feature vectors for the training data 
        for idx in range(len(train_data)):
            tokens = train_data.text[idx]
            for word, weight in tokenize(tokens):
                doc_mat[idx][word2idx[word]] += weight
        # output the vector for training data
        with open(args.index+'_train.csv', 'w') as f_out:
            for idx, feat_vect in enumerate(doc_mat):
                f_out.write(','.join(map(str, feat_vect)) + str(train_data.target[idx])+ '\n')
    
    N_vocab = len(vocabulary) + 1
    # generate the feature vectors for test file 
    with open(args.data + 'test.csv', 'r') as f:
        doc_mat = []
        test_data = pd.read_csv(f)
        for idx in range(len(test_data)):
            tokens = test_data.text[idx]
            doc_mat.append([0 for _ in range(N_vocab)])
            for word, weight in tokenize(tokens):
                if word in vocabulary:
                    doc_mat[idx][word2idx[word]] += weight
                else:
                    doc_mat[idx][-1] += 1
        with open(args.index+'_test.csv', 'w') as f_out:
            for idx, feat_vect in enumerate(doc_mat):
                f_out.write(','.join(map(str, feat_vect)) + '\n')
                
    # output the generated dictionary 
    with open(args.dict_dir+'dict.csv', 'w') as f_out:
        for idx, word in enumerate(vocabulary):
            f_out.write(str(word) + "," + str(idx) + '\n')
            

            
            
# TODO
# 1. the source files have more that one columns, choose the content that need to be processing 
# 2. combine the code for dictionary generation and feature vectors generation as they both need to open the source files--done two things simultaneously 
# 3. using the embedding as the feature vectors and apply models like FFNN or LSTM, needs to deal with the padding problems
    

    

if __name__ == '__main__':
    args = get_args()
    main(args)