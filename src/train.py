import argparse
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split
from model import myDataset

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RndFC
from sklearn.neural_network import MLPClassifier as ffp
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
from sklearn import metrics as met

# preprocessing 
# define the model 

# read the data 




def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')

    # Add data arguments
    parser.add_argument('--train_data', default='./data/index_train.csv', help='path to the BOW for training data')
    parser.add_argument('--test_data', default='./data/index_test.csv', help='path to the BOW for test data')
    parser.add_argument('--dict', default='./data/dict.csv', help='path to directory')


    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    return args

def read_from_csv(file:str):
    targets = []
    embed = []
    with open(file, 'r') as f:
        for line in f.readlines():
            tmp = list(map(int, (line.strip('\n').split(','))))
            embed.append(tmp[:-1])
            targets.append(tmp[-1])
    
    # return torch.tensor(embed), torch.tensor(targets)
    return embed, targets

def train(args):
    train_file = args.train_data
    test_file = args.test_data
    
    # generate the training data
    print('-Loading training data')
    x_data, y_data = read_from_csv(train_file)
    
    # split the whole dataset into two parts: train and validation 
    train_size = int(len(x_data) * 0.8)
    validate_size = int(len(x_data) * 0.2)
    
    # Initialize the dataset 
    # my_data = myDataset(x_data, y_data)
    # # get the corresponding data 
    # train_data, validate_data = random_split(my_data, [train_size, validate_size])
    
    # Criterion for training 
    # lossfunc = torch.nn.MSELoss()
    # model = torch.nn.SVM
    # for epch in args.nums_epochs:
    #     train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
    #     train_loss = 0.0
    #     model.train()
    
    # model = svm.SVC(C=1000)
    model = RndFC(n_estimators=50,oob_score=True,max_depth=80)
    # model = KNN()
    
    print('-Training')
    model.fit(x_data[:train_size], y_data[:train_size])
    
    print('-Predicting')
    estimates = model.predict(x_data[train_size:])
    accuracy = met.accuracy_score(y_data[train_size:],estimates)
    print('-----------Accuracy: %f' % (accuracy))
    
def eval(args):
    pass
    
    
    # model
    # criterion 
    # dataloader
    # schedular 
    

def evaluate():
    pass

def save():
    pass

if __name__ == '__main__':
    args = get_args()
    train(args)
    