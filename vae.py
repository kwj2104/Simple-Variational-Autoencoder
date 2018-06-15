import argparse
import numpy as npy
import os
from utils_vae import sigmoid, lrelu, tanh, img_tile, mnist_reader, relu, BCE_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--layersize", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--e", type=float, default=1e-8)
    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()
cpu_enabled = 0
try:
    import cupy as np
    cpu_enabled = 1
except ImportError:
    import numpy as np
    print("CuPy not enabled on this machine")
    

np.random.seed(111)

class VAE():
    def __init__(self, numbers):
        
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
        # init encoder weights
        self.e_W0 = np.random.randn(784, self.layersize).astype(np.float32) * np.sqrt(2.0/(784))
        self.e_b0 = np.zeros(self.layersize).astype(np.float32)

        self.e_W_mu = np.random.randn(self.layersize, self.nz).astype(np.float32) * np.sqrt(2.0/(self.layersize))
        self.e_b_mu = np.zeros(self.nz).astype(np.float32)
        
        self.e_W_logvar = np.random.randn(self.layersize, self.nz).astype(np.float32) * np.sqrt(2.0/(self.layersize))
        self.e_b_logvar = np.zeros(self.nz).astype(np.float32)

        # init decoder weights 
        self.d_W0 = np.random.randn(self.nz, self.layersize).astype(np.float32) * np.sqrt(2.0/(self.nz))
        self.d_b0 = np.zeros(self.layersize).astype(np.float32)
        
        self.d_W1 = np.random.randn(self.layersize, 784).astype(np.float32) * np.sqrt(2.0/(self.layersize))
        self.d_b1 = np.zeros(784).astype(np.float32)
             
        # init sample
        self.sample_z = 0
        self.rand_sample = 0
        
        # init Adam optimizer
        self.b1 = args.b1
        self.b2 = args.b2
        self.e = args.e
        self.m = [0] * 10
        self.v = [0] * 10
        self.t = 0
        
    def encoder(self, img):
        #self.e_logvar : log variance 
        #self.e_mean : mean

        self.e_input = np.reshape(img, (self.batch_size,-1))
    
        self.e_h0_l = self.e_input.dot(self.e_W0) + self.e_b0
        self.e_h0_a = lrelu(self.e_h0_l)
    		
        self.e_logvar = self.e_h0_a.dot(self.e_W_logvar) + self.e_b_logvar
        self.e_mu = self.e_h0_a.dot(self.e_W_mu) + self.e_b_mu
    
        return self.e_mu, self.e_logvar
    
    def decoder(self, z):
        #self.d_out : reconstruction image 28x28
		
        self.z = np.reshape(z, (self.batch_size, self.nz))
        
        self.d_h0_l = self.z.dot(self.d_W0) + self.d_b0		
        self.d_h0_a = relu(self.d_h0_l)

        self.d_h1_l = self.d_h0_a.dot(self.d_W1) + self.d_b1
        self.d_h1_a = sigmoid(self.d_h1_l)

        self.d_out = np.reshape(self.d_h1_a, (self.batch_size, 28, 28, 1))

        return self.d_out
    
    def forward(self, x):
        #Encode
        mu, logvar = self.encoder(x)
        
        #use reparameterization trick to sample from gaussian
        self.rand_sample = np.random.standard_normal(size=(self.batch_size, self.nz))
        self.sample_z = mu + np.exp(logvar * .5) * np.random.standard_normal(size=(self.batch_size, self.nz))
        
        decode = self.decoder(self.sample_z)
        
        return decode, mu, logvar
    
    def backward(self, x, out):
        ########################################
        #Calculate gradients from reconstruction
        ########################################
        y = np.reshape(x, (self.batch_size, -1))
        out = np.reshape(out, (self.batch_size, -1))
        
        #Calculate decoder gradients
        #Left side term
        dL_l = -y * (1 / out)
        dsig = sigmoid(self.d_h1_l, derivative=True)
        dL_dsig_l = dL_l * dsig
        
        drelu = relu(self.d_h0_l, derivative=True)

        dW1_d_l = np.matmul(np.expand_dims(self.d_h0_a, axis=-1), np.expand_dims(dL_dsig_l, axis=1))
        db1_d_l = dL_dsig_l 
        
        db0_d_l = dL_dsig_l.dot(self.d_W1.T) * drelu
        dW0_d_l = np.matmul(np.expand_dims(self.sample_z, axis=-1), np.expand_dims(db0_d_l, axis=1))
        
        #Right side term
        dL_r = (1 - y) * (1 / (1 - out))
        dL_dsig_r = dL_r * dsig
        
        dW1_d_r = np.matmul(np.expand_dims(self.d_h0_a, axis=-1), np.expand_dims(dL_dsig_r, axis=1))
        db1_d_r = dL_dsig_r
        
        db0_d_r = dL_dsig_r.dot(self.d_W1.T) * drelu
        dW0_d_r = np.matmul(np.expand_dims(self.sample_z, axis=-1), np.expand_dims(db0_d_r, axis=1))
        
        # Combine gradients for decoder
        grad_d_W0 = dW0_d_l + dW0_d_r
        grad_d_b0 = db0_d_l + db0_d_r
        grad_d_W1 = dW1_d_l + dW1_d_r
        grad_d_b1 = db1_d_l + db1_d_r
         
        #Calculate encoder gradients from reconstruction
        #Left side term
        d_b_mu_l  = db0_d_l.dot(self.d_W0.T)
        d_W_mu_l = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_mu_l, axis=1))
        
        db0_e_l = d_b_mu_l.dot(self.e_W_mu.T) * lrelu(self.e_h0_l, derivative=True)
        dW0_e_l = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_l, axis=1)) 
        
        d_b_logvar_l = d_b_mu_l * np.exp(self.e_logvar * .5) * .5 * self.rand_sample
        d_W_logvar_l = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_logvar_l, axis=1))
        
        db0_e_l_2 = d_b_logvar_l.dot(self.e_W_logvar.T) * lrelu(self.e_h0_l, derivative=True)
        dW0_e_l_2 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_l_2, axis=1)) 
        
        #Right side term
        d_b_mu_r  = db0_d_r.dot(self.d_W0.T)
        d_W_mu_r = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_mu_r, axis=1))
        
        db0_e_r = d_b_mu_r.dot(self.e_W_mu.T) * lrelu(self.e_h0_l, derivative=True)
        dW0_e_r = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_r, axis=1)) 
        
        d_b_logvar_r = d_b_mu_r * np.exp(self.e_logvar * .5) * .5 * self.rand_sample
        d_W_logvar_r = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_logvar_r, axis=1))
        
        db0_e_r_2 = d_b_logvar_r.dot(self.e_W_logvar.T) * lrelu(self.e_h0_l, derivative=True)
        dW0_e_r_2 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_r_2, axis=1))
        
        ########################################
        #Calculate encoder gradients from K-L
        ########################################
    
        #logvar terms
        dKL_b_log = -.5 * (1 - np.exp(self.e_logvar))
        dKL_W_log = np.matmul(np.expand_dims(self.e_h0_a, axis= -1), np.expand_dims(dKL_b_log, axis= 1))
        
        #Heaviside step function
        dlrelu = lrelu(self.e_h0_l, derivative=True)  

        dKL_e_b0_1 = .5 * dlrelu * (np.exp(self.e_logvar) - 1).dot(self.e_W_logvar.T)
        dKL_e_W0_1 = np.matmul(np.expand_dims(y, axis= -1), np.expand_dims(dKL_e_b0_1, axis= 1))
        
        #m^2 term
        dKL_W_m = .5 * (2 * np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(self.e_mu, axis=1)))
        dKL_b_m = .5 * (2 * self.e_mu)
        
        dKL_e_b0_2 = .5 * dlrelu * (2 * self.e_mu).dot(self.e_W_mu.T)
        dKL_e_W0_2 = np.matmul(np.expand_dims(y, axis= -1), np.expand_dims(dKL_e_b0_2, axis= 1))
        
        # Combine gradients for encoder from recon and KL
        grad_b_logvar = dKL_b_log + d_b_logvar_l + d_b_logvar_r
        grad_W_logvar = dKL_W_log + d_W_logvar_l + d_W_logvar_r
        grad_b_mu = dKL_b_m + d_b_mu_l + d_b_mu_r
        grad_W_mu = dKL_W_m + d_W_mu_l + d_W_mu_r
        grad_e_b0 = dKL_e_b0_1 + dKL_e_b0_2 + db0_e_l + db0_e_l_2 + db0_e_r + db0_e_r_2
        grad_e_W0 = dKL_e_W0_1 + dKL_e_W0_2 + dW0_e_l + dW0_e_l_2 + dW0_e_r + dW0_e_r_2
        
        
        grad_list = [grad_e_W0, grad_e_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar,
                     grad_d_W0, grad_d_b0, grad_d_W1, grad_d_b1]
        
        ########################################
        #Calculate update using Adam
        ########################################
        self.t += 1
        for i, grad in enumerate(grad_list):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.power(grad, 2)
            m_h = self.m[i] / (1 - (self.b1 ** self.t))
            v_h = self.v[i] / (1 - (self.b2 ** self.t))
            grad_list[i] = m_h / (np.sqrt(v_h) + self.e)
        
        # Update all weights
        for idx in range(self.batch_size):
            # Encoder Weights
            self.e_W0 = self.e_W0 - self.learning_rate*grad_list[0][idx]
            self.e_b0 = self.e_b0 - self.learning_rate*grad_list[1][idx]
    
            self.e_W_mu = self.e_W_mu - self.learning_rate*grad_list[2][idx]
            self.e_b_mu = self.e_b_mu - self.learning_rate*grad_list[3][idx]
            
            self.e_W_logvar = self.e_W_logvar - self.learning_rate*grad_list[4][idx]
            self.e_b_logvar = self.e_b_logvar - self.learning_rate*grad_list[5][idx]
    
            # Decoder Weights
            self.d_W0 = self.d_W0 - self.learning_rate*grad_list[6][idx]
            self.d_b0 = self.d_b0 - self.learning_rate*grad_list[7][idx]
            
            self.d_W1 = self.d_W1 - self.learning_rate*grad_list[8][idx]
            self.d_b1 = self.d_b1 - self.learning_rate*grad_list[9][idx]
    
    def train(self):
        
        #Read in training data
        trainX, _, train_size = mnist_reader(self.numbers)
        
        np.random.shuffle(trainX)
        
        #set batch indices
        batch_idx = train_size//self.batch_size
        
        total_loss = 0
        total_kl = 0
        total = 0
        
        for epoch in range(self.epochs):
            for idx in range(batch_idx):
                # prepare batch and input vector z
                train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
                #ignore batch if there are insufficient elements 
                if train_batch.shape[0] != self.batch_size:
                    break
                
                ################################
                #		Forward Pass
                ################################
                
                out, mu, logvar = self.forward(train_batch)
                
                # Reconstruction Loss
                rec_loss = BCE_loss(out, train_batch)
                
                #K-L Divergence
                kl = -0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar))
                
                loss = rec_loss + kl
                loss = loss / self.batch_size
                
                #Loss Recordkeeping
                total_loss += rec_loss / self.batch_size
                total_kl += kl / self.batch_size
                total += 1

                ################################
                #		Backward Pass
                ################################
                # for every result in the batch
                # calculate gradient and update the weights using Adam
                self.backward(train_batch, out)	

                self.img = np.squeeze(out, axis=3) * 2 - 1

                print("Epoch [%d] Step [%d]  RC Loss:%.4f  KL Loss:%.4f  lr: %.4f"%(
                        epoch, idx, rec_loss / self.batch_size, kl / self.batch_size, self.learning_rate))
                
            if cpu_enabled == 1:
                sample = np.array(self.img)
            else: 
                sample = np.asnumpy(self.img)
            
            #save image result every epoch
            img_tile(sample, self.img_path, epoch, idx, "res", True)


if __name__ == '__main__':

    # Adjust the numbers that appear in the training data. Less numbers helps 
    # run the program to see faster results
    numbers = [1, 2, 3]
    model = VAE(numbers)
    model.train()
    