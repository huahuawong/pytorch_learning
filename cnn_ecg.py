import pandas as pd
import numpy as np
import os
import json
import time
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

# Check Torch version, currently using 1.6.0 in this project file
print(torch.__version__)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = "./ecg data"


# Since we are looking into ECG arrthymia dataset, we would have to build a custom data pipeline
def load_ECG_dataset(path):
    csv_train_path = os.path.join(path, "mitbih_train.csv")
    csv_test_path = os.path.join(path, "mitbih_test.csv")
    csv_train_data = pd.read_csv(csv_train_path)
    csv_test_data = pd.read_csv(csv_test_path)

    # Split the dataset into train and test, the last column of the array has the label
    train_x = np.array(csv_train_data.iloc[:, :187], dtype=np.float32).reshape(-1, 187, 1)
    train_y = np.array(csv_train_data.iloc[:, 187], dtype=np.int32)

    test_x = np.array(csv_test_data.iloc[:, :187], dtype=np.float32).reshape(-1, 187, 1)
    test_y = np.array(csv_test_data.iloc[:, 187], dtype=np.int32)

    Nonectopic_beat, Supraventricular_ectopic_beat, Ventricular_ectopic_beat, Fusion_beat, Unknown_beat = [], [], [], [], []
    for data, label in zip(train_x, train_y):
        if label == 0:
            Nonectopic_beat.append([data, label])
        elif label == 1:
            Supraventricular_ectopic_beat.append([data, label])
        elif label == 2:
            Ventricular_ectopic_beat.append([data, label])
        elif label == 3:
            Fusion_beat.append([data, label])
        elif label == 4:
            Unknown_beat.append([data, label])

    train_dataset_dict = {"N": Nonectopic_beat, "S": Supraventricular_ectopic_beat,
                          "V": Ventricular_ectopic_beat, "F": Fusion_beat, "Q": Unknown_beat}

    print("---Training dataset---")
    print("Nonectopic beat               :", len(Nonectopic_beat))
    print("Supraventricular ectopic beat :", len(Supraventricular_ectopic_beat))
    print("Ventricular ectopic beat      :", len(Ventricular_ectopic_beat))
    print("Fusion beat                   :", len(Fusion_beat))
    print("Unknown beat                  :", len(Unknown_beat))
    print("----------------------")

    Nonectopic_beat, Supraventricular_ectopic_beat, Ventricular_ectopic_beat, Fusion_beat, Unknown_beat = [], [], [], [], []
    for data, label in zip(test_x, test_y):
        if label == 0:
            Nonectopic_beat.append([data, label])
        elif label == 1:
            Supraventricular_ectopic_beat.append([data, label])
        elif label == 2:
            Ventricular_ectopic_beat.append([data, label])
        elif label == 3:
            Fusion_beat.append([data, label])
        elif label == 4:
            Unknown_beat.append([data, label])

    test_dataset_dict = {"N": Nonectopic_beat, "S": Supraventricular_ectopic_beat,
                         "V": Ventricular_ectopic_beat, "F": Fusion_beat, "Q": Unknown_beat}

    print("---Test dataset---")
    print("Nonectopic beat               :", len(Nonectopic_beat))
    print("Supraventricular ectopic beat :", len(Supraventricular_ectopic_beat))
    print("Ventricular ectopic beat      :", len(Ventricular_ectopic_beat))
    print("Fusion beat                   :", len(Fusion_beat))
    print("Unknown beat                  :", len(Unknown_beat))
    print("----------------------")

    return train_dataset_dict, test_dataset_dict


def split_dataset(train_dataset_dict, val_num, seed=0):
    Nonectopic = train_dataset_dict["N"]
    Supraventricular = train_dataset_dict["S"]
    Ventricular = train_dataset_dict["V"]
    Fusion = train_dataset_dict["F"]
    Unknown = train_dataset_dict["Q"]

    train, validation = [], []

    np.random.seed(seed)
    np.random.shuffle(Nonectopic)
    np.random.shuffle(Supraventricular)
    np.random.shuffle(Ventricular)
    np.random.shuffle(Fusion)
    np.random.shuffle(Unknown)

    dataset_list = [Nonectopic, Supraventricular, Ventricular, Fusion, Unknown]
    for i, dataset in enumerate(dataset_list):
        for data, label in dataset[: val_num]:
            validation.append([data, label])

        for data, label in dataset[val_num:]:
            train.append([data, label])

    print("train :", len(train))
    print("validation :", len(validation))

    return train, validation

# You may notice that this is an imbalanced dataset with a majority of dataset belonging to one class, i.e.
# the non-ectopic beat
train_dataset_dict, test_dataset_dict = load_ECG_dataset(path)
train_dataset, validation_dataset = split_dataset(train_dataset_dict, val_num=100, seed=0)


# Model
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

def save_json(log, savepath):
    with open(os.path.join(savepath, "log.json"), "w") as f:
        json.dump(log, f, indent=4)

# Early stopping will be used to prevent the model from overfitting
def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, device,
                num_epochs=25, early_stopping_epoch=10, savedirpath="result"):
    os.makedirs(savedirpath, exist_ok=True)

    since = time.time()
    log_list = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping_count = 0
    start_time = time.time()
    print('TRAINING starts')
    for epoch in range(num_epochs):
        epoch = epoch + 1
        print('-' * 70)
        print('epoch : {}'.format(epoch))

        epoch_result = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_time = time.time() - start_time
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = np.float64(running_corrects.double() / len(dataloaders_dict[phase].dataset))

            if phase == "train":
                print("train      loss: {:.4f}, accuracy : {:.4f}, elapsed time: {:.4f}"
                      .format(epoch_loss, epoch_acc, epoch_time))
            else:
                print("validation loss: {:.4f}, accuracy : {:.4f}, elapsed time: {:.4f}"
                      .format(epoch_loss, epoch_acc, epoch_time))

            # deep copy the model, such that you always save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                save_model_weight_path = os.path.join(savedirpath, "trained_model.pt")
                torch.save(model.state_dict(), save_model_weight_path)

                early_stopping_count = 0
            elif phase == "val" and epoch_acc <= best_acc:
                early_stopping_count += 1

            epoch_result["epoch"] = epoch
            epoch_result["elapsed_time"] = epoch_time
            if phase == "train":
                epoch_result["train/loss"] = epoch_loss
                epoch_result["train/accuracy"] = epoch_acc

            else:
                epoch_result["validation/loss"] = epoch_loss
                epoch_result["validation/accuracy"] = epoch_acc

        log_list.append(epoch_result)
        save_json(log_list, savedirpath)

        # early stopping
        if early_stopping_count == early_stopping_epoch:
            print("Eearly stopping have been performed in this training")
            print("Epoch : {}".format(epoch))
            break

    time_elapsed = time.time() - since
    print("---------------------------------------------")
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best epoch   : {}".format(best_epoch))
    print('Best validation Accuracy: {:4f}'.format(best_acc))
    print("---------------------------------------------")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



num_epochs = 20; stopping_epoch = 5; learning_rate = 1e-3

input_size = 1; hidden_size = 128; num_layers = 1; batch_size = 128
num_classes = 5
out = "./ecg results"

# Define the data loader for train and validation dataset
train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
                        validation_dataset, batch_size=16, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# Training with the LSTM Model
model = LSTM(num_classes, input_size, hidden_size, num_layers, device)
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

trained_model = train_model(model, dataloaders_dict, criterion,
                            optimizer, exp_lr_scheduler, device,
                            num_epochs, stopping_epoch,
                            savedirpath=out)
