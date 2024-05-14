import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam

import NAE_utils

# Lets set up the early stopper!! 
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

def plot_sse_scores(model, test_loader, anomalies_tensor, device, ae = True):
    
    # Now lets check some reconstructions
    # Fetch one batch of test data
    dataiter = iter(test_loader)
    test_histograms = next(dataiter)

    # Generate reconstructions
    with torch.no_grad():
        reconstructed_histograms_sse = model.forward(test_histograms.to(device))
        reconstructed_anomalies_sse  = model.forward(anomalies_tensor.to(device))

    sse_nominal   = reconstructed_histograms_sse.detach().cpu().numpy()
    sse_anomalies = reconstructed_anomalies_sse.detach().cpu().numpy()

    sse_threshold = np.percentile(np.nan_to_num(sse_nominal, nan=1e3), 95)

    # Now the plotting part !!
    plt.figure(figsize=(12, 6))
    bins = np.linspace(min(sse_nominal), sse_threshold * 2.00, 100)
    plt.hist(sse_nominal, bins=bins, histtype='step', linewidth=2, color='blue', label='Nominal')
    plt.hist(sse_anomalies, bins=bins, histtype='step', linewidth=2, color='red', label='Anomalies')
    plt.axvline(x=sse_threshold, color='black', linestyle='--', label=f'95% threshold - Anomaly rejection: {np.mean(sse_anomalies > sse_threshold):.2f}')
    plt.title('Sum of Squared Errors (SSE) over Test and Anomalous Sets')
    plt.xlabel('SSE')
    plt.ylabel('Frequency')
    plt.legend()
    if(ae):
        plt.savefig('plots/normalized_autoencoders/AE_SSE_curve.png')
    else:
        plt.savefig('plots/normalized_autoencoders/NAE_SSE_curve.png')

    # return the percentage of anomalies that are above the threshold!
    return np.mean(sse_anomalies > sse_threshold)

def plot_loss(train_losses, test_losses, rejection_rate, ae = True):
        
        # Plotting the training and test losses
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(test_losses, label='Test Loss', color='green')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')
        #ax1.set_yscale('log')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.plot(rejection_rate, label='Rejection Rate', color='red')
        ax2.set_ylabel('Rejection Rate')
        ax2.tick_params(axis='y')
        ax2.legend()

        plt.title('Loss Progression over Epochs')
        plt.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        if(ae):
            plt.savefig('plots/normalized_autoencoders/Loss_ae_curve.png')
        else:
            plt.savefig('plots/normalized_autoencoders/Loss_nae_curve.png')

def train_a_nae(train_loader, test_loader, anomalies_tensor):
    
    # Lets first define a encoder and a decoder!
    model = NAE_utils.NAE()
    #print(model)
    
    # Now, lets train this bad boy here !!
    
    # We have the ae and the nae, the ae is the warm-up phase
    # After it converges, we can train the nae!! This is on the paper!
    n_ae_epoch  = 100
    n_nae_epoch = 50
    
    ae_opt = Adam(model.parameters(), lr=1e-3)
    
    # Should this have any difference??
    l_params = [{'params': list(model.encoder.parameters()) + list(model.decoder.parameters())}]
    nae_opt = Adam(l_params, lr=1e-5)
    # I should check about this trainable temperature there ...
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Early stop AE
    early_Stopper_ae = EarlyStopper( patience = 5 )
    
    train_loss_ae, test_loss_ae, anomaly_rejection_ae = [], [], []
    
    # First we train the AE
    model.train()
    for i_epoch in range(n_ae_epoch):
        running_loss = 0.0
        for data in train_loader:
            
            # This already perform the backpropagation and the optimization step
            d_train = model.train_step_ae(data.to(device), ae_opt, clip_grad=0.1)
            running_loss += d_train['loss']
        print( f"Iter [{i_epoch:d}] Avg Loss: {(running_loss/len(train_loader)):.4f} ")
        #print(f"Iter [{i_epoch:d}] Avg Loss: {d_train['loss']:.4f} ")
        with torch.no_grad():
            model.eval()
            running_test_loss = 0.0
            for x in test_loader:
                d_test = model.train_step_ae(x.to(device), ae_opt, validation = True) #model.validation_step(x.to(device))
                running_test_loss += d_test
            print('Test Loss: ', running_test_loss/len(test_loader))
        # Storing the losses!
        train_loss_ae.append(running_loss/len(train_loader))
        test_loss_ae.append(running_test_loss/len(test_loader))
        
        # Now lets check the performance of the AE in the sse plots !!
        #plot_sse_scores(model, test_loader, anomalies_tensor, device, ae = True)  
        
        anomaly_rejection_ae.append(plot_sse_scores(model, test_loader, anomalies_tensor, device, ae = True))
           
        # Now lets plot it !!   
        plot_loss(train_loss_ae, test_loss_ae, anomaly_rejection_ae,  ae = True)    
            
        # Save the model's state_dict
        torch.save(model.state_dict(), f'saved_models/nae/model_ae_{i_epoch}.pth')
            
        if( early_Stopper_ae.early_stop( running_test_loss/len(test_loader) ) ):
            best_epoch = np.argmin(test_loss_ae)
            model.load_state_dict(torch.load(f'saved_models/nae/model_ae_{best_epoch}.pth'))
            break  
                
    # In principle, the model is already trained, now we can train the NAE
    train_loss_nae, test_loss_nae, anomaly_rejection_nae = [], [], []
    
    # Now setting up the nae early stopper !!
    early_Stopper_nae = EarlyStopper( patience = 5 )
    
    print('\n\n Now the NAE training!!')
    for i_epoch in range(n_nae_epoch):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            model.train()
            d_result = model.train_step(data.to(device), nae_opt)
            running_loss += d_result['loss']
        # Now the test loss:
        model.eval()
        running_test_loss = 0.0
        for x in test_loader:
            d_test = model.train_step(x.to(device), nae_opt, validation = True)
            running_test_loss += d_test
                        
        print( f"Iter [{i_epoch:d}] Test Avg Loss: {(running_loss/len(train_loader)):.4f} ")
        print( 'Test Loss: ', running_test_loss/len(test_loader))
        
        # Storing the losses!
        train_loss_nae.append(running_loss/len(train_loader))
        test_loss_nae.append(running_test_loss/len(test_loader))
    
        anomaly_rejection_nae.append(plot_sse_scores(model, test_loader, anomalies_tensor, device, ae = False))    
    
        # Now lets plot it !!   
        plot_loss(train_loss_nae, test_loss_nae, anomaly_rejection_nae,  ae = False)      

        # Save the model's state_dict
        torch.save(model.state_dict(), f'saved_models/nae/model_nae_{i_epoch}.pth')
            
        if( early_Stopper_nae.early_stop( running_test_loss/len(test_loader) ) ):
            best_epoch = np.argmin(test_loss_nae)
            model.load_state_dict(torch.load(f'saved_models/nae/model_nae_{best_epoch}.pth'))
            
            # Exit the training loop
            break

    # Now lets check the performance of the NAE in the sse plots !!
    #plot_sse_scores(model, test_loader, anomalies_tensor, device, ae = False)  
    
    """
    TODO:
    - Check the loss functions 
    - Check the probabilities
    - Check the sampled samples xD
    - Check the ROC curve vs AE !!!
        - We can use it own AE to make a unbiased estimative
    """
    
    # Now lets check some reconstructions of this bad boy here!
    with torch.no_grad():
        model.eval()
        ii = 0
        for x in test_loader:
            reconstructed = model.reconstruct(x.to(device)).detach().cpu()
        
            # Now, lets make some simple plots of these bad boyzz
            num_examples = min(3, len(reconstructed))
            plt.figure(figsize=(14, num_examples * 4))
            for i in range(num_examples):
                #sse = np.sum((original[i] - reconstructed[i]) ** 2)
                plt.subplot(num_examples, 1, i + 1)
                plt.plot(x[i].detach().numpy(), label='Original', color='blue', linewidth=3)
                plt.plot(reconstructed[i].detach().numpy(), label='Reconstructed', color='red', alpha=0.7, linewidth=3)
                #plt.title(f"{title} {i + 1} - SSE: {sse:.2e}")
                plt.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig(f'plots/normalized_autoencoders/nae_{ii}_histograms.png')
            plt.close()
            
            ii += 1
    
    
    
    return 0