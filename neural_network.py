import numpy
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from functions import external_data_dir
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from prepare_data import get_balanced_dataset
from sklearn import preprocessing


class UNSWClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(UNSWClassifier, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.l1.bias)
        # self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.l2.bias)
        # self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.l3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.l3.bias)
        self.relu3 = nn.ReLU()
        nn.init.kaiming_normal_(self.l3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.l3.bias)

    def forward(self, x):
        x = self.l1(x)
        # x = self.relu1(x)
        x = self.l2(x)
        # x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        return x


def bool_to_int(x):
    if (isinstance(x, (bool))):
        if x == True:
            return 1
        else:
            return 0
    else:
        return x


def prepare_dataframe(size):
    ds = pd.read_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv', sep=",", header=0,
                     # converters={'ct_ftp_cmd': empty_string_to_nan, 'is_ftp_login': empty_string_to_nan,
                     #           'ct_flw_http_mthd': empty_string_to_nan},
                     encoding='utf8')

    ds = get_balanced_dataset(ds, size)

    enc = OneHotEncoder(handle_unknown='ignore')
    y_df = pd.DataFrame(enc.fit_transform(ds[['attack_cat']]).toarray())

    """le = preprocessing.LabelEncoder()
    y_fitted = le.fit_transform(ds.attack_cat)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    y_inverted = le.inverse_transform(y_fitted)
    ds.attack_cat = y_fitted"""

    ds = ds.replace({True: 1, False: 0})
    x = ds[[b for b in list(ds.columns) if b not in ['level_0', 'attack_cat', 'label', 'index']]]
    x = x.copy()
    print('x.head')
    print(x.head())
    # y= ds.attack_cat

    x.to_csv(external_data_dir() + '/' + 'raw_data_x_nn.csv', index=False)
    #ds.attack_cat.to_frame().to_csv(external_data_dir() + '/' + 'raw_data_y_nn.csv', index=False)
    y_df.to_csv(external_data_dir() + '/' + 'raw_data_y_nn.csv', index=False)


class MyDataSet(Dataset):
    def __init__(self):
        x = np.loadtxt(external_data_dir() + '/' + 'raw_data_x_nn.csv', delimiter=",", dtype=np.float32, skiprows=1,
                       encoding="UTF-8", max_rows=10000, ndmin=2)


        y = np.loadtxt(external_data_dir() + '/' + 'raw_data_y_nn.csv', delimiter=",",
                       skiprows=1, dtype=np.float32, encoding="UTF-8", max_rows=10000)

        self.x = torch.from_numpy(x.astype(np.float32))

        self.y = torch.from_numpy(y.astype(np.int64))
        self.y = self.y.type(torch.FloatTensor)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def get_dataset():
    dataset = MyDataSet()
    return dataset


def train(model, device, train_loader, optimizer, epoch):
    print('training')
    i = 1
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"batch number {i}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        print('criterion')
        print(loss)
        # loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        i = i+1
        print(f"num of items in batch:{len(target)}")


def cross_validate(dataset):
    # Define the number of folds and batch size
    num_items = len(dataset)
    k_folds = 5
    batch_size = 20

    print('num_items')
    print(num_items)
    # items_per_fold = num_items/k_folds
    num_items_for_training = (num_items * (k_folds - 1)) / k_folds
    print('num_items_for_training')
    print(num_items_for_training)
    num_items_per_training_batch = num_items_for_training / batch_size
    print('num_items_per_training_batch')
    print(num_items_per_training_batch)
    if not num_items_per_training_batch.is_integer():
        return
    num_batches = num_items_for_training / num_items_per_training_batch
    print('num_batches')
    print(num_batches)
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print('len_train_idx')
        print(len(train_idx))
        print('len_test_idx')
        print(len(test_idx))
        print(f"Fold {fold + 1}")
        print("-------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                  sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                                  )
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                 sampler=torch.utils.data.SubsetRandomSampler(test_idx),
                                 )

        # Initialize the model and optimizer
        model = UNSWClassifier(input_size=63, num_classes=10).to(device)

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        # Train the model on the current fold
        for epoch in range(1, int(num_batches)):
            train(model, device, train_loader, optimizer, epoch)

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                print('data')
                print(data)
                print('target')
                print(target)
                data, target = data.to(device), target.to(device)
                output = model(data)
                print('output')
                print(output)
                test_loss = criterion(output, target)
                # test_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
                # only working when not using onehotencoding for targets
                #pred = output.argmax(dim=1, keepdim=True)
                #correct += pred.eq(target.view_as(pred)).sum().item()
        # only working when not using onehotencoding for targets
        #test_loss /= len(test_loader.dataset)
        #accuracy = 100.0 * correct / len(test_loader.dataset)

        # Print the results for the current fold
        # only working when not using onehotencoding for targets
        #print(
        #    f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


def go_neural1():
    prepare_dataframe(2000)
    dataset = get_dataset()
    cross_validate(dataset)

    # num_epochs = 2


def go_neural(raw_data):
    print('**************************')

    # 0) Prepare data
    # bc = datasets.load_breast_cancer()
    X = balanced_data.drop('attack_cat', axis=1)
    y = balanced_data.attack_cat.copy()

    # tensor_X = torch.tensor(X.values)
    # tensor_y = torch.tensor(y_fitted)

    n_samples, n_features = X.shape
    print(X.shape)
    print(X.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y_fitted, test_size=0.2, random_state=1234)
    print(y_train)
    # scale
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    X_train = torch.from_numpy(X_train.astype(np.float32).to_numpy())
    X_test = torch.from_numpy(X_test.astype(np.float32).to_numpy())
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.type(torch.LongTensor)
    y_test = y_test.type(torch.LongTensor)

    print('y_train tensor')
    print(type(y_train))
    # y_train = y_train.view(y_train.shape[0], 1)
    # y_test = y_test.view(y_test.shape[0], 1)

    # 1) Model
    # Linear model f = wx + b , sigmoid at the end

    model = UNSWClassifier(input_size=64, num_classes=10)
    print(model)

    # 2) Loss and optimizer
    num_epochs = 10000
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3) Training loop
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        y_predicted_cls_1d = torch.argmax(y_predicted_cls, dim=1)
        print('shapes')
        print(y_predicted_cls_1d.shape)
        print(y_test.shape)
        acc = y_predicted_cls_1d.eq(y_test).sum() / float(y_test.shape[0])
        print(f'accuracy: {acc.item():.4f}')
