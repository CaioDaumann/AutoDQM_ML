import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim*(0.75) )),
            nn.ReLU(True),
            nn.Linear( int(input_dim*(0.75)) , int(input_dim*(0.5) )),
            nn.ReLU(True),
            nn.Linear( int(input_dim*(0.5)), int(input_dim*(0.25))),
            nn.ReLU(True),
            nn.Linear(  int(input_dim*(0.25)), encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, int(input_dim*(0.25) )),
            nn.ReLU(True),
            nn.Linear( int(input_dim*(0.25)) , int(input_dim*(0.5) )),
            nn.ReLU(True),
            nn.Linear( int(input_dim*(0.5)), int(input_dim*(0.75))),
            nn.ReLU(True),
            nn.Linear( int(input_dim*(0.75)), int(input_dim))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class train_AE:
    def __init__(self,train_loader,test_loader, anomalies_tensor, input_size, encoding_dim = 16, epochs = 100, global_mean = 0, global_std = 1, linear = True):    
    
        self.Is_Linear = linear
        if(self.Is_Linear):
            self.model_ae = Autoencoder( input_dim = input_size, encoding_dim = encoding_dim)
        else:
            self.model_ae = ConvAutoencoder( input_channels = 1, encoding_dim = encoding_dim)
    
        self.train_loader     = train_loader
        self.test_loader      = test_loader
        self.anomalies_tensor = anomalies_tensor
        
        self.epochs = epochs
    
        self.global_mean = global_mean
        self.global_std = global_std
    
    def train_AE(self ):
    
        # Loss function
        criterion = torch.nn.MSELoss()

        # Optimizer (You can adjust the learning rate as needed)
        optimizer = optim.AdamW(self.model_ae.parameters(), lr=5e-4, weight_decay=1e-7) #Lets add some L2 regularization

        early_stopper = EarlyStopper(patience=15)
        
        # Lists to store loss values
        self.train_losses = []
        self.test_losses = []
        
        # Training and Validation Loop
        for epoch in range(self.epochs):
            self.model_ae.train()  # Set the model to training mode
            train_loss = 0.0
            for data in self.train_loader:
                inputs = data
                           
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model_ae(inputs)
                loss = criterion(outputs, inputs)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track the loss
                train_loss += loss.item()

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Evaluation Loop
            self.model_ae.eval()  # Set the model to evaluation mode
            test_loss = 0.0
            with torch.no_grad():
                for data in self.test_loader:
                    inputs = data
                
                    outputs = self.model_ae(inputs)
                    loss = criterion(outputs, inputs)
                    
                    test_loss += loss.item()
                    
            if( early_stopper.early_stop(test_loss) ):
                print("Early Stopping!")
                break

            # Calculate average test loss for the epoch
            avg_test_loss = test_loss / len(self.test_loader)
            self.test_losses.append(avg_test_loss)

            # Print training and test loss
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}')

        # After training is over lets evaluate the results
        self.plot_loss()
        self.evaluate_model()
        #self.compute_SSE_curve()
        #return self.model_ae
        
    def plot_loss(self):
        
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Progression over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.savefig('plots/autoencoders/Loss.png')
        
    def evaluate_model(self):
        # Ensure the model is in evaluation mode
        self.model_ae.eval()

        # Fetch one batch of test data
        dataiter = iter(self.test_loader)
        test_histograms = next(dataiter)

        # Generate reconstructions
        with torch.no_grad():
            reconstructed_histograms = self.model_ae(test_histograms)
            reconstructed_anomalies = self.model_ae(self.anomalies_tensor)

        # Unstandardize the histograms
        test_histograms_unstd           = test_histograms * self.global_std + self.global_mean
        reconstructed_histograms_unstd  = reconstructed_histograms * self.global_std + self.global_mean
        reconstructed_anomalies_unstd   = reconstructed_anomalies * self.global_std + self.global_mean
        anomalies_tensor_unstd          = self.anomalies_tensor * self.global_std + self.global_mean

        # Convert tensors to numpy for plotting
        test_histograms_unstd = test_histograms_unstd.cpu().numpy()
        reconstructed_histograms_unstd = reconstructed_histograms_unstd.cpu().numpy()
        reconstructed_anomalies_unstd = reconstructed_anomalies_unstd.cpu().numpy()
        anomalies_tensor_unstd = anomalies_tensor_unstd.cpu().numpy()

        # Lets normalize everything to one!
        #test_histograms_unstd = test_histograms_unstd / np.sum(test_histograms_unstd, axis=1, keepdims=True)
        #reconstructed_histograms_unstd = reconstructed_histograms_unstd / np.sum(reconstructed_histograms_unstd, axis=1, keepdims=True)
        #reconstructed_anomalies_unstd = reconstructed_anomalies_unstd / np.sum(reconstructed_anomalies_unstd, axis=1, keepdims=True)
        #anomalies_tensor_unstd = anomalies_tensor_unstd / np.sum(anomalies_tensor_unstd, axis=1, keepdims=True)
        
        # Calculate and plot SSE distributions
        self.plot_sse_distributions(test_histograms_unstd, reconstructed_histograms_unstd,
                                    anomalies_tensor_unstd, reconstructed_anomalies_unstd)

        # Plot original vs. reconstructed histograms
        self.plot_histograms(test_histograms_unstd, reconstructed_histograms_unstd, title='Nominal')
        self.plot_histograms(anomalies_tensor_unstd, reconstructed_anomalies_unstd, title='Anomalous')

        # Plot histograms with high and low SSE
        self.plot_extreme_sse_histograms(test_histograms_unstd, reconstructed_histograms_unstd, 'Nominal')
        self.plot_extreme_sse_histograms(anomalies_tensor_unstd, reconstructed_anomalies_unstd, 'Anomalous')

    def plot_sse_distributions(self, test_histograms, reconstructed_histograms, anomalies, reconstructed_anomalies):
        sse_nominal = np.sum((test_histograms - reconstructed_histograms) ** 2, axis=1)
        sse_anomalies = np.sum((anomalies - reconstructed_anomalies) ** 2, axis=1)

        sse_threshold = np.percentile(np.nan_to_num(sse_nominal, nan=1e3), 95)

        #print(sse_nominal)
        #print(sse_threshold)
        #print(sse_anomalies)
        #exit()

        plt.figure(figsize=(12, 6))
        bins = np.linspace(min(sse_nominal), sse_threshold * 1.10, 100)
        plt.hist(sse_nominal, bins=bins, histtype='step', linewidth=2, color='blue', label='Nominal')
        plt.hist(sse_anomalies, bins=bins, histtype='step', linewidth=2, color='red', label='Anomalies')
        plt.axvline(x=sse_threshold, color='black', linestyle='--', label=f'95% threshold - Anomaly rejection: {np.mean(sse_anomalies > sse_threshold):.2f}')
        plt.title('Sum of Squared Errors (SSE) over Test and Anomalous Sets')
        plt.xlabel('SSE')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('plots/autoencoders/SSE_curve.png')
        plt.close()

    def plot_histograms(self, original, reconstructed, title):
        num_examples = min(5, len(original))
        plt.figure(figsize=(14, num_examples * 4))
        for i in range(num_examples):
            sse = np.sum((original[i] - reconstructed[i]) ** 2)
            plt.subplot(num_examples, 1, i + 1)
            plt.plot(original[i], label='Original', color='blue', linewidth=3)
            plt.plot(reconstructed[i], label='Reconstructed', color='red', alpha=0.7, linewidth=3)
            plt.title(f"{title} {i + 1} - SSE: {sse:.2e}")
            plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f'plots/autoencoders/{title.replace(" ", "_").lower()}_histograms.png')
        plt.close()

    def plot_extreme_sse_histograms(self, original, reconstructed, set_name):
        sse = np.sum((original - reconstructed) ** 2, axis=1)
        high_sse_indices = sse >= np.percentile(sse, 95)
        low_sse_indices = sse <= np.percentile(sse, 5)

        self.plot_histograms(original[high_sse_indices], reconstructed[high_sse_indices], title=f'{set_name} High SSE')
        self.plot_histograms(original[low_sse_indices], reconstructed[low_sse_indices], title=f'{set_name} Low SSE')

    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Transposed conv output size: output_size=(input_size−1)×stride+kernel_size−2×padding+output_padding   
class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, encoding_dim):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # Output: (200 + 2 - 3) / 2 + 1 = 100
            nn.Tanh(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),  # Output: (100 + 2 - 3) / 2 + 1 = 50
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=3, stride=2, padding=1)  # Output: (50 + 2 - 3) / 2 + 1 = 25

        )

        # Linear layers in latent space
        self.encoder_linear = nn.Linear(25, encoding_dim)
        self.latent_linear = nn.Linear(encoding_dim, encoding_dim)
        self.decoder_linear = nn.Linear(encoding_dim, 25)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Input: 25 -> Output: 50
            nn.Tanh(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Input: 50 -> Output: 100
            nn.Tanh(),
            nn.ConvTranspose1d(32, input_channels, kernel_size=2, stride=2, padding=0, output_padding=0)  # Input: 100 -> Output: 200
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the last conv layer
        x = self.encoder_linear(x)
        x = F.tanh(x)
        x = self.latent_linear(x)
        x = F.tanh(x)
        x = self.decoder_linear(x)
        x = F.tanh(x)
        x = x.view(x.size(0), 1, -1)  # Reshape back to match the expected input of the decoder
        x = self.decoder(x)
        x = x.flatten(1)
        return x

    
    
class train_ConvAE:
    def __init__(self,train_loader,test_loader, input_size, encoding_dim = 16, epochs = 100):    
    
 
        self.model_ae = ConvAutoencoder( input_channels = 1, encoding_dim = encoding_dim)
    
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.epochs = epochs
    
    def train_AE(self ):
    
        # Loss function
        criterion = torch.nn.MSELoss()

        # Optimizer (You can adjust the learning rate as needed)
        optimizer = optim.AdamW(self.model_ae.parameters(), lr=1e-3, weight_decay=1e-7) #Lets add some L2 regularization

        early_stopper = EarlyStopper(patience=15)
        
        # Lists to store loss values
        self.train_losses = []
        self.test_losses = []
        
        # Training and Validation Loop
        for epoch in range(self.epochs):
            self.model_ae.train()  # Set the model to training mode
            train_loss = 0.0
            for data in self.train_loader:
                inputs = data
                       

                inputs = inputs.unsqueeze(1)     
                 
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model_ae(inputs)
                loss = criterion(outputs, inputs.flatten(1))
                
                # Backward pass and optimize
                loss.backward()
                
                # lets clip the gradients - this clips the gradients after backpropagation, so the big gradient is backpropagated
                # I should find a way to do it during the backpropagation
                torch.nn.utils.clip_grad_norm_(self.model_ae.parameters(), 1.0)
                
                optimizer.step()
                
                # Track the loss
                train_loss += loss.item()

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Evaluation Loop
            self.model_ae.eval()  # Set the model to evaluation mode
            test_loss = 0.0
            with torch.no_grad():
                for data in self.test_loader:
                    inputs = data
                    
                    inputs = inputs.unsqueeze(1)
                    outputs = self.model_ae(inputs)
                    
                    loss = criterion(outputs, inputs.flatten(1))
                    test_loss += loss.item()
                    
            if( early_stopper.early_stop(test_loss) ):
                print("Early Stopping!")
                break

            # Calculate average test loss for the epoch
            avg_test_loss = test_loss / len(self.test_loader)
            self.test_losses.append(avg_test_loss)

            # Print training and test loss
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}')

        # After training is over lets evaluate the results
        self.plot_loss()
        self.evaluate_model()
        #self.compute_SSE_curve()
        
        #return self.model_ae
        
    def plot_loss(self):
        
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Progression over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.savefig('Loss_Conv.png')
        
    def evaluate_model(self):
        # Plotting the reconstructed histograms!

        # Assuming `model_ae` is already trained and `self.test_loader` is defined
        self.model_ae.eval()  # Ensure the model is in evaluation mode

        # Fetch one batch of test data
        dataiter = iter(self.test_loader)
        test_histograms = next(dataiter)

        test_histograms = test_histograms.unsqueeze(1)

        # Generate reconstructions
        with torch.no_grad():
            reconstructed_histograms = self.model_ae(test_histograms)
            test_histograms = test_histograms.flatten(1)

        # Convert tensors to numpy for plotting
        test_histograms = test_histograms.cpu().numpy()
        reconstructed_histograms = reconstructed_histograms.cpu().numpy()

        # Plotting the SSE curve
        sse = np.sum((test_histograms - reconstructed_histograms) ** 2,1)
        plt.figure(figsize=(12, 6))
        bins = np.linspace( np.mean(sse) - 3*np.std(sse) , np.mean(sse) + 3*np.std(sse), 50)
        plt.hist(sse, bins = bins)
        plt.title('Sum of Squared Errors (SSE) over Test Set')
        plt.xlabel('SSE')
        plt.ylabel('Events')
        #plt.yscale('log')
        plt.savefig('SSE_curve_Conv.png')
        plt.close()

        # Plot original and reconstructed histograms
        # Plot original and reconstructed histograms and calculate SSE
        num_examples = min(4, len(test_histograms))  # Plot up to 5 examples
        plt.figure(figsize=(14, num_examples * 4))
        for i in range(num_examples):
            # Calculate SSE
            sse = np.sum(( test_histograms[i]/np.sum(test_histograms[i]) - reconstructed_histograms[i]/np.sum(reconstructed_histograms[i])) ** 2)
            
            # Plotting
            plt.subplot(num_examples, 1, i + 1)
            plt.plot(test_histograms[i], label='Original', color='blue', linewidth = 3)
            plt.plot(reconstructed_histograms[i], label='Reconstructed', color='red', alpha=0.7, linewidth = 3)
            plt.title(f"Original vs. Reconstructed Histogram {i+1} - SSE: {sse:.4e}")
            plt.legend(fontsize=16)

        plt.tight_layout()
        plt.savefig('reconstructed_Histograms_Conv.png')
        #print('Number of training examples:', len(self.train_tensor), ' and Testing examples:', len(self.test_tensor))