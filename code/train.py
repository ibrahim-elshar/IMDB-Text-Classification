import torch
import torch.nn as nn
import torch.optim as optim
from models import lstm as lstm
from data import dataloader

from tqdm import tqdm

args = None

def get_accuracy(logits, labels):
    rounded_preds = torch.round(torch.sigmoid(logits))
    correct = (rounded_preds == labels).float() 
    acc = correct.sum()/len(correct)
    return acc



def train(model, data, optimizer, criterion):
    global args
    running_loss = 0.0
    total_loss = 0.0
    total_acc = 0.0

    model.train()
    i=0
    for batch in tqdm(data):
        optimizer.zero_grad()
        pred = model(batch.text).squeeze(1)
        loss = criterion(pred, batch.label)
        acc = get_accuracy(pred, batch.label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        total_acc += acc.item()

        if i % args.log_interval == args.log_interval-1:
            print(f"Running Loss : {running_loss/args.log_interval}")
            running_loss = 0.0
        i+=1


    total_loss /= len(data)
    total_acc /= len(data)
    print(f"Epoch Loss: {total_loss}, Epoch Accuracy: {total_acc}")
    return total_loss, total_acc


def validate(model, data, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for i,batch in enumerate(data):
        pred = model(batch.text).squeeze(1)
        loss = criterion(pred, batch.label)
        acc = get_accuracy(pred, batch.label)
        total_loss += loss.item()
        total_acc += acc.item()
    total_loss /= len(data)
    total_acc /= len(data)
    return total_loss, total_acc




def main(args):
    train_loader, val_loader, test_loader, glove_vecs =\
                             dataloader.get_data(args.batch_size)

    if args.checkpoint is not False:
        model = torch.load(args.checkpoint)
    else:
        if args.model == 'lstm':
            model = lstm.LSTM(pretrained_emb=glove_vecs)
        else:
            raise NotImplemented
    

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, 
                                        verbose=True)

    prev_best = -1
    for epoch in range(args.epochs):
        loss, acc = train(model, train_loader, optimizer, criterion)
        vloss, vacc = validate(model, val_loader, criterion)
        lr_scheduler.step(vloss)
        print(f"Validation Loss: {vloss}, Validation Accuracy: {vacc}")
        # checkpoint
        if vacc > prev_best:
            prev_best = vacc
            torch.save(model, 
                f"models/checkpoints/{args.run_name}-epoch-{epoch}-acc-{vacc}")
            print(f"Saving Checkpoint: models/checkpoints/{args.run_name}-epoch-{epoch}-acc-{vacc}")
    tloss, tacc = validate(model, test_loader, criterion)
    print(f"Test Loss: {tloss}, Test Accuracy: {tacc}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('--model', default='lstm')
    parser.add_argument('--batch_size', default=45)
    parser.add_argument('--checkpoint', default=False)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--log_interval', default=50)

    args = parser.parse_args()

    main(args)