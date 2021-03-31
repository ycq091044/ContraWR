import torch
from utils import SHHSLoader, SLEEPCALoader
import numpy as np
import torch.nn as nn
import os
from collections import Counter
import argparse
from sklearn.metrics import confusion_matrix
from model import CNNEncoder2D_SHHS, CNNEncoder2D_SLEEP
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--lr', type=float, default=2e-4, help="learning rate")
parser.add_argument('--percent', type=float, default=100, help="train percent")
parser.add_argument('--n_dim', type=int, default=128, help="hidden units")
parser.add_argument('--dataset', type=str, default='SHHS', help="dataset")
args = parser.parse_args()

def train(model, optimizer, Epoch, loss_func, train_loader, test_loader):

    model.train()
    acc_list = []

    # train
    for epoch in range(Epoch):
        print ()
        # Train
        correct_train, total_train, loss_train = [], [], []
        max_min = [0,0]
        for idx, (X_train, y_train) in enumerate(tqdm(train_loader)):
            X_train, y_train = X_train, y_train.flatten()
            X_train, y_train = X_train.to(device), y_train.to(device)

            # backpropagation
            optimizer.zero_grad()
            pred = model(X_train, sup=True)
            loss = loss_func(pred, y_train)
            loss.backward()
            optimizer.step()
            
            # print ('compute accuracy')
            total_train.append(y_train.shape[0])
            correct_train.append((torch.argmax(pred.data, 1) == y_train).sum().item())
            loss_train.append(loss.item())
            
        print ("epoch: {}/{}, avg_loss: {}, train accuracy: {:.2f}%".format(epoch, \
            args.epochs, sum(loss_train) / len(loss_train), sum(correct_train) / sum(total_train) * 100))

        # evaluation
        with torch.no_grad():
            model.eval()
            
            pred, target = [], []
            for idx, (X_test, y_test) in enumerate(test_loader):
                X_test, y_test = X_test, y_test.flatten()
                X_test, y_test = X_test.to(device), y_test

                y_pred = torch.argmax(model(X_test, sup=True), 1).data.cpu()
                pred += list(y_pred); target += list(y_test.numpy())

            correct_test = (np.array(pred) == np.array(target)).sum()
            total_test = len(pred)
            print ("------------------------------")
            print ("epoch: {}/{}, train accuracy: {:.2f}%, test accuracy: {:.2f}%".format(epoch, \
                args.epochs, sum(correct_train) / sum(total_train) * 100, correct_test / total_test * 100))
            
            acc_list.append(correct_test / total_test)
            # print (confusion_matrix(pred, target))

            model.train()
        
        if epoch > 10:
            print ('recent five epoch, mean: {}, std: {}'.format(np.mean(acc_list[-10:]), np.std(acc_list[-10:])))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    seed = 1234
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    # dataset
    if args.dataset == 'SHHS':

        # load data
        pretext_dir = './SHHS_data/processed/pretext/'
        train_dir = './SHHS_data/processed/train/'
        test_dir = './SHHS_data/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        train_index = os.listdir(train_dir)
        train_index = train_index[:len(train_index)//10]
        test_index = os.listdir(test_dir)
        test_index = test_index[:len(test_index)//10]
        
        print ('pretext (all patient): ', len(pretext_index))
        print ('train (all patient): ', len(train_index))
        print ('test (all patient): ', len(test_index))

        # data loader
        pretext_loader = torch.utils.data.DataLoader(SHHSLoader(pretext_index, pretext_dir, False), 
                        batch_size=128, shuffle=True, num_workers=10)
        train_loader = torch.utils.data.DataLoader(SHHSLoader(train_index, train_dir, False), 
                        batch_size=128, shuffle=False, num_workers=10)
        test_loader = torch.utils.data.DataLoader(SHHSLoader(test_index, test_dir, False), 
                        batch_size=128, shuffle=False, num_workers=10)

        # define model
        model = CNNEncoder2D_SHHS(args.n_dim)
        model.to(device)

    elif args.dataset == 'SLEEP':

        # load data
        pretext_dir = './SLEEP_data/cassette_processed/pretext/'
        train_dir = './SLEEP_data/cassette_processed/train/'
        test_dir = './SLEEP_data/cassette_processed/test/'

        pretext_index = os.listdir(pretext_dir)
        train_index = os.listdir(train_dir)
        test_index = os.listdir(test_dir)
        
        print ('pretext (all patient): ', len(pretext_index))
        print ('train (all patient): ', len(train_index))
        print ('test (all patient): ', len(test_index))

        # data loader
        pretext_loader = torch.utils.data.DataLoader(SLEEPCALoader(pretext_index, pretext_dir, False), 
                        batch_size=128, shuffle=True, num_workers=10)
        train_loader = torch.utils.data.DataLoader(SLEEPCALoader(train_index, train_dir, False), 
                        batch_size=128, shuffle=False, num_workers=10)
        test_loader = torch.utils.data.DataLoader(SLEEPCALoader(test_index, test_dir, False), 
                        batch_size=128, shuffle=False, num_workers=10)

        # define model
        model = CNNEncoder2D_SLEEP(args.n_dim)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # training
    train(model, optimizer, args.epochs, loss_func, train_loader, test_loader)
