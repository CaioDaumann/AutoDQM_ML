import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import zuko

from utils import EarlyStopper

# Lets write a class to perform the morphing of the 1d histograms into a gaussian!!
# A interesting thing to test here is also use a feature extractor, and train the flow in these features!

"""
So, there are some ways to do it!

- First and easiest, morph the data (histogram) into the latent space!
- Second: Use these feature extractors and apply the flows in these features! (latent space)
    - But, how do we train this feature extractor?? Is it like a autoencoder?? Then we take only the encoder?
        - They used something called AlexNet! (read more about it)
        - This is almost like contrastive learning, right??  
        - It is a average! 
    - Link to the paper (https://arxiv.org/pdf/2008.12577)
- I should read something about this thing where they say that the data is defined in a low dimensional manifold or something like this!    

ANOTHER INTERESTING IDEIA OF THE PAPER:
- since we keep downscaling the data with the convolutions ex 28x28 -> 14x14 -> 7x7 -> 3x3
- we can feed each one of these  14x14, 7x7, 3x3 into the flow and then combine the results in the end!
- But how are these results combined?

- perhaps we can even do bin by bin anomaly dtecetionw ith the flows by looking into the gaussianish at each bin!

"""

class normalizing_flow:
    def __init__(self, train_loader, test_loader,anomalies_tensor, n_flow_layers, n_hidden_features, n_hidden, input_dim):
        
        # This should work for 1d histograms for now
        self.n_flow_layers = n_flow_layers
        self.n_hidden_features = n_hidden_features
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.flow = zuko.flows.NSF(features=input_dim, transforms=n_flow_layers, hidden_features=[n_hidden_features]*n_hidden, passes=2).to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.anomalies_tensor = anomalies_tensor.to(self.device)

        self.epochs = 100
    
        self.train_flow()
        self.plot_likelihood()
        self.plot_loss()
        self.plot_latent_space()
    
    def train_flow(self):
        
        # Optimizer (You can adjust the learning rate as needed)
        optimizer = optim.AdamW(self.flow.parameters(), lr=5e-4, weight_decay=1e-8) #Lets add some L2 regularization

        early_stopper = EarlyStopper(patience=15)
        
        # Lists to store loss values
        self.train_losses = []
        self.test_losses = []
        
        # Training and Validation Loop
        for epoch in range(self.epochs):
            self.flow.train()  # Set the model to training mode
            train_loss = 0.0
            for data in self.train_loader:
                inputs = data.to(self.device)
                           
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                loss = -self.flow().log_prob(inputs).mean()

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track the loss
                train_loss += loss.item()

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Evaluation Loop
            self.flow.eval()  # Set the model to evaluation mode
            test_loss = 0.0
            with torch.no_grad():
                for data in self.test_loader:
                    inputs = data.to(self.device)
            
                    loss =  -self.flow().log_prob(inputs).mean()        
                    test_loss += loss.item()
                    
            if( early_stopper.early_stop(test_loss) ):
                print("Early Stopping!")
                break

            # Calculate average test loss for the epoch
            avg_test_loss = test_loss / len(self.test_loader)
            self.test_losses.append(avg_test_loss)

            # Print training and test loss
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}')
            
            # lets do per epoch validation!!
            self.plot_likelihood()
            self.plot_loss()
            self.plot_latent_space()
    
    def plot_loss(self):
        
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Progression over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/Flow_loss.png')
    
    def plot_latent_space(self):
        
        latent_space_100 = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs = data.to(self.device)
                latent_space_100.append(self.flow().transform(inputs).detach().cpu().numpy())
        # Lets concatenate everything
        latent_space_100 = np.concatenate([ array for array in latent_space_100 ]) 

        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        bins = np.linspace(-4, 4, 50)
        plt.hist(latent_space_100, bins = bins, histtype='step' , label='Test Loss')
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/base_distribution_flows.png')
        
        # Plotting only the gaussians close to the center!!!
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        bins = np.linspace(-4, 4, 50)
        plt.hist(latent_space_100[92:108], bins = bins, histtype='step' , label='Test Loss')
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/base_distribution_center_flows.png')
      
      
    def log_prob_standard_normal(self,x):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)  # Add a new axis if X is a 1D array
            log_probs = (x**2)/2 
            return np.sum(log_probs, axis =1)/len(log_probs)   
    # Lets plot the likelihood for the test set!
    def plot_likelihood(self):
        
        likelihoods, likelihoods_anomalies = [],[]
        gauss_log_prob, gaus_log_prob_anomalies = [],[]
        with torch.no_grad():
            for data in self.test_loader:
                inputs = data.to(self.device)
            
                # Lets produce some anomalies !!
                #inputs = inputs + 0.5*torch.randn_like(inputs)
                 
                likelihoods.append(-self.flow().log_prob(inputs).detach().cpu().numpy())  
                gauss_log_prob.append(self.log_prob_standard_normal(self.flow().transform(inputs).detach().cpu().numpy() ))
        
        # Now calculating the log prob for the anomalies
        likelihoods_anomalies.append(-self.flow().log_prob(self.anomalies_tensor).detach().cpu().numpy())  
        gaus_log_prob_anomalies.append(self.log_prob_standard_normal(self.flow().transform(self.anomalies_tensor).detach().cpu().numpy() ))
        
        # Lets concatenate everything
        likelihoods = np.concatenate([ array for array in likelihoods ])
        gauss_log_prob = np.concatenate([ array for array in gauss_log_prob ])
        gaus_log_prob_anomalies = np.concatenate([ array for array in gaus_log_prob_anomalies ])
        likelihoods_anomalies = np.concatenate([ array for array in likelihoods_anomalies ])
        
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        plt.hist(likelihoods           , bins = 100, label=f'- log Likelihood - mean: {np.mean(likelihoods)} ')
        plt.hist(likelihoods_anomalies , bins = 100, label=f'- log Likelihood - mean: {np.mean(likelihoods_anomalies)} ')
        plt.title('Log likelihood for the test set')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/log_likelihood_flows.png')
        
        # Now the exponentiaded!
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        bins = np.linspace(0, 0.15, 70)
        plt.hist(gauss_log_prob          , bins = bins, histtype='step', linewidth = 2 , label=f'- log Likelihood - mean: {np.mean(gauss_log_prob)} ')
        plt.hist(gaus_log_prob_anomalies , bins = bins, histtype='step', linewidth = 2 , color = 'red', label=f'- log Likelihood - mean: {np.mean(gaus_log_prob_anomalies)} ')
        
        sorted_gauss_log_prob = np.sort(gauss_log_prob)
        gauss_log_prob_treshold = sorted_gauss_log_prob[int(0.95*len(sorted_gauss_log_prob))]
        
        # Lets calculate the anomaly rejection rate
        print(f'Anomaly rejection rate: {np.sum(gaus_log_prob_anomalies > gauss_log_prob_treshold)/len(gaus_log_prob_anomalies)}')
        
        plt.axvline(x=gauss_log_prob_treshold, color='black', linestyle='--', label =  f'95 threshold - Anomaly rejection: {np.sum(gaus_log_prob_anomalies > gauss_log_prob_treshold)/len(gaus_log_prob_anomalies)}')
        
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/exp_likelihood_flows.png')
            
                
########################################################################################################               
# This flow is based on the output of a networks that has the objective of perform a feature extraction!
########################################################################################################
class feature_extraction_flow:
    def __init__(self, train_loader, test_loader, n_flow_layers, n_hidden_features, n_hidden, input_dim):
        
        # This should work for 1d histograms for now
        self.n_flow_layers = n_flow_layers
        self.n_hidden_features = n_hidden_features
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    
        self.epochs = 5
    
        self.define_model()
        self.train_flow()
        self.plot_likelihood()
        self.plot_loss()
        self.plot_latent_space()
    
    def define_model(self):
        
        # Lets define the feature extractor
        self.feature_extractor = nn.Sequential(
            
            nn.Linear(self.input_dim,  int(self.input_dim/2) ),
            nn.ReLU(),
            nn.Linear(int(self.input_dim/2), int(self.input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.input_dim/4), int(self.input_dim/10))
            
        ).to(self.device)
        
        
        self.flow = zuko.flows.NSF(features=int(self.input_dim/10), transforms=self.n_flow_layers, hidden_features=[self.n_hidden_features]*self.n_hidden, passes=2).to(self.device)

    
    def train_flow(self):
        
        # Get parameters from both modules
        parameters = list(self.flow.parameters()) + list(self.feature_extractor.parameters())
        
        # Optimizer (You can adjust the learning rate as needed)
        optimizer = optim.AdamW(parameters, lr=5e-4, weight_decay=1e-12) #Lets add some L2 regularization

        early_stopper = EarlyStopper(patience=15)
        
        # Lists to store loss values
        self.train_losses = []
        self.test_losses = []
        
        # Training and Validation Loop
        for epoch in range(self.epochs):
            self.flow.train()  # Set the model to training mode
            train_loss = 0.0
            for data in self.train_loader:
                inputs = data.to(self.device)
                           
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                features = self.feature_extractor(inputs)
                # Forward pass
                loss = -self.flow().log_prob(features).mean()

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track the loss
                train_loss += loss.item()

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Evaluation Loop
            self.flow.eval()  # Set the model to evaluation mode
            test_loss = 0.0
            with torch.no_grad():
                for data in self.test_loader:
                    inputs = data.to(self.device)
                    features = self.feature_extractor(inputs)
            
                    loss =  -self.flow().log_prob(features).mean()        
                    test_loss += loss.item()
                    
            if( early_stopper.early_stop(test_loss) ):
                print("Early Stopping!")
                break

            # Calculate average test loss for the epoch
            avg_test_loss = test_loss / len(self.test_loader)
            self.test_losses.append(avg_test_loss)

            # Print training and test loss
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}')
            
            # lets do per epoch validation!!
            self.plot_likelihood()
            self.plot_loss()
            self.plot_latent_space()
    
    def plot_loss(self):
        
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Progression over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/feature_flow/Flow_loss.png')
    
    def plot_latent_space(self):
        
        latent_space_100 = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs = data.to(self.device)
                features = self.feature_extractor(inputs)
                
                latent_space_100.append(self.flow().transform(features).detach().cpu().numpy())
        # Lets concatenate everything
        latent_space_100 = np.concatenate([ array for array in latent_space_100 ]) 

        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        bins = np.linspace(-4, 4, 50)
        plt.hist(latent_space_100, bins = bins, histtype='step' , label='Test Loss')
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/feature_flow/base_distribution_flows.png')
        
      
    def log_prob_standard_normal(self,x):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)  # Add a new axis if X is a 1D array
            log_probs = (x**2)/2 
            return np.sum(log_probs, axis =1)      
      
    # Lets plot the likelihood for the test set!
    def plot_likelihood(self):
        
        likelihoods = []
        gauss_log_prob = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs = data.to(self.device)
                
                features = self.feature_extractor(inputs)
                
                likelihoods.append(-self.flow().log_prob(features).detach().cpu().numpy())  
                gauss_log_prob.append(self.log_prob_standard_normal(self.flow().transform(features).detach().cpu().numpy() ))
        
        # Lets concatenate everything
        likelihoods = np.concatenate([ array for array in likelihoods ])
        gauss_log_prob = np.concatenate([ array for array in gauss_log_prob ])
        
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        plt.hist(likelihoods, bins = 100, label=f'- log Likelihood - mean: {np.mean(likelihoods)} ')
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/feature_flow/log_likelihood_flows.png')
        
        # Now the exponentiaded!
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        plt.hist(gauss_log_prob, bins = 100, label=f'- log Likelihood - mean: {np.mean(gauss_log_prob)} ')
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/feature_flow/exp_likelihood_flows.png')
        
        # Only the log prob !!!
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        #plt.plot(self.train_losses, label='Train Loss')
        plt.hist(  np.clip(np.exp(-likelihoods),a_min = 0,a_max = 1e6) , bins = 100, label=f'- log Likelihood - mean: {np.mean(likelihoods)} ')
        plt.title('Log likelihood for the test set')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        plt.legend()
        #plt.yscale('log')
        plt.savefig('plots/feature_flow/likelihood_flows.png')

            