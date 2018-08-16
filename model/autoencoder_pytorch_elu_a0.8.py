import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import torch.utils.data as Data

data = pd.read_csv('good_data_starts_valley.csv')
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler((0,1))
data_normalized = sc.fit_transform(data.T).T
from sklearn.model_selection import train_test_split
random_state = 11
training_set, test_set = train_test_split(data_normalized, test_size = 0.2, random_state = random_state)
nb_diff = data.shape[1]
nb_train = training_set.shape[0]
nb_test = test_set.shape[0]

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
torch_dataset = Data.TensorDataset(training_set,training_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

BATCH_SIZE = 256
loader = Data.DataLoader(
    dataset = torch_dataset,      # torch TensorDataset format
    batch_size = BATCH_SIZE,      # mini batch size
    shuffle = True,               # random shuffle for training            # subprocesses for loading data
)

EPOCH = 10000
LR = 0.001      # learning rate

ALPHA = 0.8

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(nb_diff, 200),
            nn.ELU(alpha = ALPHA),
            nn.Linear(200, 100),
            nn.ELU(alpha = ALPHA),
            nn.Linear(100, 50),
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ELU(alpha = ALPHA),
            nn.Linear(100, 200),
            nn.ELU(alpha = ALPHA),
            nn.Linear(200, nb_diff),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

#optimize for cuda
autoencoder.cuda()
loss_func.cuda()


    
for epoch in range(EPOCH):
    train_loss = 0
    s = 0.
    start_time = time.time()
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = batch_x.view(-1, 1 * 1500)
        b_x_cuda = b_x.cuda()
        b_y = batch_y.view(-1, 1 * 1500)
        b_y_cuda = b_y.cuda()
        encoded, decoded = autoencoder(b_x_cuda)
        decoded_cpu = decoded.cpu()
        optimizer.zero_grad()
        loss = loss_func(decoded, b_y_cuda)
        loss.backward()
        train_loss += np.sqrt(loss.data[0])
        s += 1.
        #decide the amount of weight to update
        optimizer.step()
    if epoch % 50 == 0 or epoch == (EPOCH - 1):
        end_time = time.time()
        dur_time = float(end_time - start_time)
        format_str = ("epoch:%s loss:%s Time Used:%.3f")
        print(format_str % (str(epoch), str(train_loss / s), dur_time))



#visualize training data
arrange = 240
plt.figure(figsize = (20,10))
plt.title('123')
c_id = 10
for p in range(1,9,1):
    arrange += 1
    c_plot = plt.subplot(arrange)
    c_input = Variable(training_set[c_id]).unsqueeze(0).cuda()
    c_encoded, c_decoded = autoencoder(c_input)
    c_decoded = c_decoded.cpu().data.numpy()
    c_plot.plot(training_set[c_id].data.numpy(), color = 'yellow')
    c_plot.plot(c_decoded[0], color = 'red')
    c_id += 1
    
#visualize test data
arrange = 240
plt.figure(figsize = (20,10))
plt.title('123')
c_id = 10
for p in range(1,9,1):
    arrange += 1
    c_plot = plt.subplot(arrange)
    c_input = Variable(test_set[c_id]).unsqueeze(0).cuda()
    c_encoded, c_decoded = autoencoder(c_input)
    c_decoded = c_decoded.cpu().data.numpy()
    c_plot.plot(test_set[c_id].data.numpy(), color = 'yellow')
    c_plot.plot(c_decoded[0], color = 'red')
    c_id += 1


#calculate all mse on training and test data
from sklearn.metrics import mean_squared_error

training_mse = 0
for c_id in range(len(training_set)):
    c_input = Variable(training_set[c_id]).unsqueeze(0).cuda()
    c_encoded, c_decoded = autoencoder(c_input)
    c_decoded = c_decoded.cpu().data.numpy()
    training_mse += mean_squared_error(training_set[c_id].data.numpy(), c_decoded[0])

training_mse = training_mse / len(training_set)



test_mse = 0
for c_id in range(len(test_set)):
    c_input = Variable(test_set[c_id]).unsqueeze(0).cuda()
    c_encoded, c_decoded = autoencoder(c_input)
    c_decoded = c_decoded.cpu().data.numpy()
    test_mse += mean_squared_error(test_set[c_id].data.numpy(), c_decoded[0])

test_mse = test_mse / len(test_set)

#calculate peak and valley loss
#peak and valley loss
import peak_loss_pytorch
training_loss2 = []
testing_loss2 = []
train_data = training_set.data.numpy()
test_data = test_set.data.numpy()
c_input_training = Variable(training_set).cuda()
c_input_testing = Variable(test_set).cuda()
c_encoded, c_decoded = autoencoder(c_input_training)
result = c_decoded.cpu().data.numpy()
training_loss2.append('%.5f'%(peak_loss_pytorch.cal_loss(result, train_data)))
c_encoded, c_decoded = autoencoder(c_input_testing)
result = c_decoded.cpu().data.numpy()
testing_loss2.append('%.5f'%(peak_loss_pytorch.cal_loss(result, test_data)))

#save model
NAME = F'autoencoder_elu_alpha_{ALPHA}'
torch.save(autoencoder.state_dict(), NAME)






        