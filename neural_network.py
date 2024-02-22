import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import get_balanced_dataset
from sklearn import preprocessing


class UNSWClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


def go_neural(raw_data):
    print('**************************')
    balanced_data = get_balanced_dataset(raw_data, 20000)
    print('size bal. data', len(balanced_data))

    print(balanced_data.attack_cat.value_counts())
    # 0) Prepare data
    #bc = datasets.load_breast_cancer()
    X = balanced_data.drop('attack_cat', axis=1)
    y = balanced_data.attack_cat.copy()



    le = preprocessing.LabelEncoder()
    #df = ["paris", "paris", "tokyo", "amsterdam"]

    y_fitted = le.fit_transform(y)
    y_inverted = le.inverse_transform(y_fitted)
    print('y_fitted')
    print(y_fitted)
    print('y_inverted')
    print(y_inverted)

    #tensor_X = torch.tensor(X.values)
    #tensor_y = torch.tensor(y_fitted)



    n_samples, n_features = X.shape
    print(X.shape)
    print(X.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y_fitted, test_size=0.2, random_state=1234)
    print(y_train)
    # scale
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)
    X_train = torch.from_numpy(X_train.astype(np.float32).to_numpy())
    X_test = torch.from_numpy(X_test.astype(np.float32).to_numpy())
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.type(torch.LongTensor)
    y_test = y_test.type(torch.LongTensor)

    print('y_train tensor')
    print(type(y_train))
    #y_train = y_train.view(y_train.shape[0], 1)
    #y_test = y_test.view(y_test.shape[0], 1)

    # 1) Model
    # Linear model f = wx + b , sigmoid at the end

    model = UNSWClassifier(input_size=64, hidden_size=16, num_classes=10)
    print(model)

    # 2) Loss and optimizer
    num_epochs = 1000
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


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
