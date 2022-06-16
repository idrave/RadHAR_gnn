from secrets import choice
import torch
import torch.nn as nn
import os
from MyDataset import LMDBDataset
from torch.utils.data import Dataset, DataLoader
import argparse
from PointGNN import HAR_PointGNN, PointGNN, MMPointGNN
from pathlib import Path
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', required=True, help="Path to data folder")
    parser.add_argument('--replace', action='store_true', help="Whether to replace existing LMDB in data directory")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers for DataLoaders")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train for")
    parser.add_argument('--log_steps', type=int, default=300, help="Train logging frequency in number of batches")
    parser.add_argument('--bs_train', type=int, default=8, help="Training batch size")
    parser.add_argument('--bs_eval', type=int, default=8, help="Evaluation batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    # Model args
    parser.add_argument('--model', choices=['pointgnn','mmpointgnn'], default='pointgnn', help="Model to be trained")
    parser.add_argument('-r', type=float, default=0.05, help="Radius for PointGNN adjacency")
    parser.add_argument('--layers', type=int, default=3, help="Number of PointGNN layers")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability")
    # Scheduler args
    parser.add_argument('--steplr_size', type=int, default=1, help="Frequency of scheduler step in epochs")
    parser.add_argument('--steplr_gamma', type=float, default=0.9, help="Learning rate decay factor")
    if args:
        return parser.parse_args(args)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    batch_size = args.bs_train
    test_batch = args.bs_eval
    learning_rate = args.lr
    frame_num = 60
    epoch_num = args.epochs
    data_path = Path(args.data_path)
    num_workers = args.num_workers
    
    dataset = LMDBDataset(str(data_path/'Train'),str(data_path/'lmdbData_train'),frame_num,replace=args.replace)
    dataset_test = LMDBDataset(str(data_path/'Test'),str(data_path/'lmdbData_test'),frame_num,replace=args.replace)
    train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(dataset = dataset_test,batch_size=test_batch,shuffle=False,num_workers=num_workers)
    print(device)
    if args.model == "pointgnn":
        gnn = PointGNN(T=args.layers, r=args.r, state_dim=8, dropout=args.dropout)
    else:
        gnn = MMPointGNN(T=args.layers, r=args.r, state_dim=8, dropout=args.dropout)
    model = HAR_PointGNN(gnn, dropout=args.dropout)
    model.to(device)

    # if os.path.exists('./models/HAR_PointGNN.pkl'):
    #     model.load_state_dict(torch.load('./models/HAR_PointGNN.pkl',map_location = device))
    #     print("load model sucessfully")

    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=args.steplr_size, gamma=args.steplr_gamma)

    crossloss = nn.CrossEntropyLoss()
    print('Starting training')
    batch_num = 1
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        start = time.time()
        for i, batch in enumerate(train_loader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            frame_sz = batch[2].to(device)
            adam.zero_grad()
            outputs = model(inputs,frame_sz)
            loss = crossloss(outputs,targets)
            loss.backward()
            adam.step()
            epoch_loss += loss
            if batch_num % args.log_steps == 0:
                print('[TRAIN {:0>5d} | {:0>3d}] loss {:.5f}'.format(
                        i + 1, epoch + 1, loss))
            batch_num += 1
        scheduler.step()
        print('[TRAIN {}] epoch loss:{:.5f}\t elapsed:{:.2f}'.format(
                        epoch + 1, epoch_loss, time.time()-start))
        # Save model
        torch.save(model.state_dict(), "./models/HAR_PointGNN.pkl")
        # Evaluate model
        model.eval()
        test_correct = 0
        for batch in test_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            frame_sz = batch[2].to(device)
            outputs = model(inputs, frame_sz)
            _, pred = torch.max(outputs, 1)
            test_correct += torch.sum(pred == targets)
        acc = 100.0*test_correct/len(test_loader)
        print('[VALID {}] accuracy:{:.3f}\t elapsed:{:.2f}'.format(
                        epoch + 1, acc, time.time()-start))

