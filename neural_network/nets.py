import torch
import torch.nn as nn


# BasicNet only contains a single hidden layer
class BasicNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BasicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class AdvancedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.0):
        super(AdvancedNet, self).__init__()
        self.nn_list = nn.ModuleList()
        self.nn_list.append(nn.Linear(input_size, hidden_sizes[0]))
        self.nn_list.append(nn.ReLU())
        if dropout:
            self.nn_list.append(nn.Dropout(p=dropout))

        for i in range(1, len(hidden_sizes)):
            self.nn_list.append(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.nn_list.append(nn.ReLU())
            if dropout:
                self.nn_list.append(nn.Dropout(p=dropout))
        self.nn_list.append(nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        for module in self.nn_list:
            x = module(x)
        return x
