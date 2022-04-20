import torch
from utils import SLEEPCALoader, SHHSLoader
import numpy as np
import torch.nn as nn
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from model import CNNEncoder2D_SLEEP, CNNEncoder2D_SHHS
from loss import MoCo, SimCLR, BYOL, OurLoss, SimSiam
from tqdm import tqdm
from collections import Counter
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# evaluation design
def task(X_train, X_test, y_train, y_test, n_classes):
            
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=500)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    
    res = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    
    return res, cm

def Pretext(q_encoder, k_encoder, optimizer, Epoch, criterion, pretext_loader, train_loader, test_loader):

    q_encoder.train(); k_encoder.train()

    global queue
    global queue_ptr
    global n_queue

    step = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

    all_loss, acc_score = [], []
    for epoch in range(Epoch):
        # save model
        # torch.save(q_encoder.state_dict(), open(os.path.join('saved', args.model, \
        #     'Epoch_{}-T-{}-delta-{}.model'.format(epoch, args.T, args.delta)), 'wb'))
        print ()
        for index, (aug1, aug2) in enumerate(tqdm(pretext_loader)):
            aug1, aug2 = aug1.to(device), aug2.to(device)
            if args.model in ['BYOL']:
                emb_aug1 = q_encoder(aug1, mid=False, byol=True)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['SimCLR']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = q_encoder(aug2, mid=False)
            elif args.model in ['ContraWR']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['MoCo']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['SimSiam']:
                emb_aug1, proj1 = q_encoder(aug1, simsiam=True)
                emb_aug2, proj2 = q_encoder(aug2, simsiam=True)

            # backpropagation
            if args.model == 'MoCo':
                loss = criterion(emb_aug1, emb_aug2, queue)
                if queue_ptr + emb_aug2.shape[0] > n_queue:
                    queue[queue_ptr:] = emb_aug2[:n_queue-queue_ptr]
                    queue[:queue_ptr+emb_aug2.shape[0]-n_queue] = emb_aug2[-(queue_ptr+emb_aug2.shape[0]-n_queue):]
                    queue_ptr = (queue_ptr + emb_aug2.shape[0]) % n_queue
                else:
                    queue[queue_ptr:queue_ptr+emb_aug2.shape[0]] = emb_aug2
            elif args.model == 'SimSiam':
                loss = criterion(proj1, proj2, emb_aug1, emb_aug2)
            else:
                loss = criterion(emb_aug1, emb_aug2)

            # loss back
            all_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # only update encoder_q

            # exponential moving average (EMA)
            for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                param_k.data = param_k.data * args.m + param_q.data * (1. - args.m) 

            N = 1000
            if (step + 1) % N == 0:
                acc_score.append([sum(all_loss[-N:]) / len(all_loss[-N:]), evaluate(q_encoder, train_loader, test_loader)])
                scheduler.step(sum(all_loss[-50:]))
            step += 1

        # print the lastest result
        print ('epoch: {}'.format(epoch))
        for i in acc_score[-10:]:
            print (i)

        if len(acc_score) >= 5:
            print ('mean: {}, std: {}'.format(np.array(acc_score)[-5:, -1].mean(), np.array(acc_score)[-5:, -1].std()))


def evaluate(q_encoder, train_loader, test_loader):

    # freeze
    q_encoder.eval()

    # process val
    emb_val, gt_val = [], []
    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)
    # print(Counter(gt_val))

    emb_test, gt_test = [], []
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test= np.array(emb_test), np.array(gt_test)
    # print(Counter(gt_test))                

    res, cm = task(emb_val, emb_test, gt_val, gt_test, 5)
    # print (cm, 'accuracy', res)
    # print (cm)
    q_encoder.train()
    return res

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.5e-3, help="learning rate")
    parser.add_argument('--n_dim', type=int, default=128, help="hidden units (for SHHS, 256, for Sleep, 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
    parser.add_argument('--training', type=int, default=10, help="training subject")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient")
    parser.add_argument('--model', type=str, default='ContraWR', help="which model")
    parser.add_argument('--T', type=float, default=0.3,  help="T")
    parser.add_argument('--sigma', type=float, default=2.0,  help="sigma")
    parser.add_argument('--delta', type=float, default=0.2,  help="delta")
    parser.add_argument('--dataset', type=str, default='SLEEP', help="dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    # set random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # torch.backends.cudnn.benchmark = True

    global queue
    global queue_ptr
    global n_queue

    if args.dataset == 'SLEEP':
        # dataset
        pretext_dir = '/srv/local/data/SLEEPEDF/cassette_processed/pretext/'
        train_dir = '/srv/local/data/SLEEPEDF/cassette_processed/train/'
        test_dir = '/srv/local/data/SLEEPEDF/cassette_processed/test/'

        pretext_index = os.listdir(pretext_dir)
        train_index = os.listdir(train_dir)
        train_index = train_index[:len(train_index)//2]
        test_index = os.listdir(test_dir)

        print ('pretext (all patient): ', len(pretext_index))
        print ('train (all patient): ', len(train_index))
        print ('test (all) patient): ', len(test_index))

        pretext_loader = torch.utils.data.DataLoader(SLEEPCALoader(pretext_index, pretext_dir, True), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SLEEPCALoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SLEEPCALoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        # define and initialize the model
        q_encoder = CNNEncoder2D_SLEEP(args.n_dim)
        q_encoder.to(device)

        k_encoder = CNNEncoder2D_SLEEP(args.n_dim)
        k_encoder.to(device)

    elif args.dataset == 'SHHS':
        # dataset
        pretext_dir = '/srv/local/data/SHHS/processed/pretext/'
        train_dir = '/srv/local/data/SHHS/processed/train/'
        test_dir = '/srv/local/data/SHHS/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        pretext_index = pretext_index[:len(pretext_index)//10]
        train_index = os.listdir(train_dir)
        train_index = train_index[:len(train_index)//10]
        test_index = os.listdir(test_dir)
        test_index = test_index[:len(test_index)//10]

        print ('pretext (all patient): ', len(pretext_index))
        print ('train (all patient): ', len(train_index))
        print ('test (all) patient): ', len(test_index))

        pretext_loader = torch.utils.data.DataLoader(SHHSLoader(pretext_index, pretext_dir, True), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SHHSLoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SHHSLoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        # define the model
        q_encoder = CNNEncoder2D_SHHS(args.n_dim)
        q_encoder.to(device)

        k_encoder = CNNEncoder2D_SHHS(args.n_dim)
        k_encoder.to(device)

    for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        param_k.data.copy_(param_q.data) 
        param_k.requires_grad = False  # not update by gradient

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # assign contrastive loss function
    if args.model == 'ContraWR':
        criterion = OurLoss(device, args.delta, args.sigma, args.T).to(device)
    elif args.model == 'MoCo':
        criterion = MoCo(device).to(device)
        queue_ptr, n_queue = 0, 4096
        queue = torch.tensor(np.random.rand(n_queue, args.n_dim), dtype=torch.float).to(device)
    elif args.model == 'SimCLR':
        criterion = SimCLR(device).to(device)
    elif args.model == 'BYOL':
        criterion = BYOL(device).to(device)
    elif args.model == 'SimSiam':
        criterion = SimSiam(device).to(device)

    # optimize
    Pretext(q_encoder, k_encoder, optimizer, args.epochs, criterion, pretext_loader, train_loader, test_loader)
