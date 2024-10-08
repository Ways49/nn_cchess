import json
import os.path
import random
import torch
import torch.nn as nn
import torch.optim as opt
from get_data import create_data_loader
from model import net,block
from convert_data import get_filepaths
from torchsummary import summary

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = net(in_channels=15,out_channels=128,num_classes=2086).to(device)
    model = nn.DataParallel(model)
    summary(model,input_size=(15,10,9))
    #
    file_list = get_filepaths("data\\test_converted_data","json")
    #random.seed(7070)
    random.shuffle(file_list)
    #
    split_spot = int(len(file_list) * 0.9)
    #
    file_list_train = file_list[:split_spot]
    file_list_eval = file_list[split_spot:]
    #
    num_workers = 10
    batch_size = 128
    data_loader_train = create_data_loader(file_list_train,batch_size,num_workers=num_workers)
    data_loader_eval = create_data_loader(file_list_eval,batch_size,num_workers=num_workers)
    #
    optimizer = opt.RAdam(params=model.parameters(),lr=3e-4)
    loss_method = nn.CrossEntropyLoss().to(device)
    for epoch in range(10000):
        print("-------------------------------------------------------------------------")
        train_loss = 0
        train_acc = 0
        batch_num = 0
        for i,data in enumerate(data_loader_train):
            data_train,data_target_idx = data
            #
            data_train[0] = torch.as_tensor(data_train[0],dtype=torch.float32).cuda()
            data_train[1] = torch.as_tensor(data_train[1],dtype=torch.float32).cuda()
            data_target_idx = torch.as_tensor(data_target_idx,dtype=torch.int64).cuda()
            #
            pred = model(data_train)
            #
            loss = loss_method(pred,data_target_idx)
            train_loss += loss.item()
            batch_num += 1
            #
            train_acc += (pred.argmax(1).to(device) == data_target_idx).sum() / len(data_target_idx)
            #
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f"epoch {epoch + 1} -> {i} | train")
        train_loss /= batch_num
        train_acc /= batch_num
        print(f"epoch {epoch + 1} -> train loss:{train_loss} | train_acc:{train_acc} ")
        #
        test_loss = 0
        test_acc = 0
        test_batch_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader_eval):
                data_test, data_test_target_idx = data
                #
                data_test[0] = torch.as_tensor(data_test[0], dtype=torch.float32).cuda()
                data_test[1] = torch.as_tensor(data_test[1], dtype=torch.float32).cuda()
                data_test_target_idx = torch.as_tensor(data_test_target_idx, dtype=torch.int64).cuda()
                #
                pred = model(data_test)
                #
                loss = loss_method(pred, data_test_target_idx)
                test_loss += loss.item()
                test_batch_num += 1
                #
                test_acc += (pred.argmax(1).to(device) == data_test_target_idx).sum() / len(data_test_target_idx)
                #print(f"epoch {epoch + 1} -> {i} | test")
            test_loss /= test_batch_num
            test_acc /= test_batch_num
            print(f"epoch {epoch + 1} -> test loss:{test_loss} | test_acc:{test_acc} ")
        #
        torch.save(model.state_dict(),os.path.join("models",f"epoch_{epoch + 1}_tcc_{int(test_acc * 1000)}.pth"))

