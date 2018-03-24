import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--layersize", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()

class VAE():
    def __init__(self):
        
        self.numbers = numbers

        self.epochs = args.epoch
        self.batch_size = args.bsize
        self.learning_rate = args.lr
        self.decay = 0.001
        self.nz = args.nz
        self.layersize = args.layersize

        self.img_path = "./images"
        if not os.path.exists(self.img_path):
                os.makedirs(self.img_path)
        
        # Xavier initialization is used to initialize the weights
        # https://theneuralperspective.com/2016/11/11/weights-initialization/
        # init encoder weights
        self.g_W0 = np.random.randn(784, self.layersize).astype(np.float32) * np.sqrt(2.0/(100))
        self.g_b0 = np.zeros(self.layersize).astype(np.float32)

        self.g_W_mu = np.random.randn(self.layersize,self.nz).astype(np.float32) * np.sqrt(2.0/(128))
        self.g_b_mu = np.zeros(self.nz).astype(np.float32)
        
        self.g_W_logvar = np.random.randn(self.layersize,self.nz).astype(np.float32) * np.sqrt(2.0/(128))
        self.g_b_logvar = np.zeros(self.nz).astype(np.float32)

        # init decoder weights 
        self.d_W0 = np.random.randn(784,128).astype(np.float32) * np.sqrt(2.0/(784))
        self.d_b0 = np.zeros(128).astype(np.float32)
        
        self.d_W1 = np.random.randn(128,1).astype(np.float32) * np.sqrt(2.0/(128))
        self.d_b1 = np.zeros(1).astype(np.float32)
        
    def encoder(self, img):
    		#self.d_h{num}_l : hidden logit layer
    		#self.d_h{num}_a : hidden activation layer
    
        self.d_input = np.reshape(img, (self.batch_size,-1))
    
        self.d_h0_l = self.d_input.dot(self.d_W0) + self.d_b0
        self.d_h0_a = lrelu(self.d_h0_l)
    		
        self.d_h1_l = self.d_h0_a.dot(self.d_W1) + self.d_b1
        self.d_h1_a = sigmoid(self.d_h1_l)
        self.d_out = self.d_h1_a
    
        return self.d_h1_l, self.d_out
    
    def decoder(self, z):
        #self.g_h{num}_l : hidden logit layer
        #self.g_h{num}_a : hidden activation layer
		
        self.z = np.reshape(z, (self.batch_size, -1))
        self.g_h0_l = self.z.dot(self.g_W0) + self.g_b0
		
        self.g_h0_a = lrelu(self.g_h0_l)

        self.g_h1_l = self.g_h0_a.dot(self.g_W1) + self.g_b1
        self.g_h1_a = tanh(self.g_h1_l)
        self.g_out = np.reshape(self.g_h1_a, (self.batch_size, 28, 28))
		
		
        return self.g_h1_l, self.g_out

if __name__ == '__main__':
    print("test")