from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np


#Print out full np arrays
np.set_printoptions(threshold=np.nan)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
             
        #Xavier Initialization
        torch.nn.init.xavier_normal(self.fc1.weight)
        torch.nn.init.xavier_normal(self.fc21.weight)
        torch.nn.init.xavier_normal(self.fc22.weight)
        torch.nn.init.xavier_normal(self.fc3.weight)
        torch.nn.init.xavier_normal(self.fc4.weight)
        torch.nn.init.constant(self.fc1.bias, 0)
        torch.nn.init.constant(self.fc21.bias, 0)
        torch.nn.init.constant(self.fc22.bias, 0)
        torch.nn.init.constant(self.fc3.bias, 0)
        torch.nn.init.constant(self.fc4.bias, 0)
        
        np.save('fc1_w.npy', self.fc1.weight.data.numpy())
        np.save('fc21_w.npy', self.fc21.weight.data.numpy())
        np.save('fc22_w.npy', self.fc22.weight.data.numpy())
        np.save('fc3_w.npy', self.fc3.weight.data.numpy())
        np.save('fc4_w.npy', self.fc4.weight.data.numpy())
        
        

    def encode(self, x):
        h1 = self.lrelu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            #eps = Variable(std.data.new(std.size()).normal_())
            #np.save('eps.npy', eps.data.numpy())
            eps = Variable(torch.from_numpy(np.load('eps.npy')))
            rep = eps.mul(std).add_(mu)

            return rep
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        
        z = self.reparameterize(mu, logvar)
        
        decode = self.decode(z)
        
        return decode, mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=.0001)
#optimizer = optim.Adam(model.parameters(), lr=.0001)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #return KLD
    #ORIGINAL LOSS
    return BCE + KLD, BCE, KLD


def train(epoch):
    counter = 0
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        #data = Variable(data)
        data = Variable(torch.from_numpy(np.load('test_batch.npy'))).squeeze(3).unsqueeze(1)

        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        

        
        loss, BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        

        
#        count = 0
#        for p in model.parameters():
#            if count == 9:
#                #print(p.size())
#                #print(p.grad[20])
#                print(p.grad[0:300])
#            count += 1

        train_loss += loss.data[0]
        
        optimizer.step()
        
        if counter == 18:
            pass
            #print(recon_batch.data.numpy())
#            for i, p in enumerate(model.parameters()):
#                if i == 1:
#                    print(p.grad)
#                    print(p.grad[30].numpy())
#                    x = p.grad.numpy()
#                    np.save('pt_sample.npy', p.grad[30].numpy())
#                    print(x[30])
                    #print(np.load('pt_sample.npy'))
        

        

                    
        print("#: {} Train Loss: {} {} ".format(counter, BCE, KLD))
        
        if counter == 20:
            raise Exception()
        counter += 1
        
#        if batch_idx % args.log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader),
#                loss.data[0] / len(data)))
            

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        loss, _, _ = loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')