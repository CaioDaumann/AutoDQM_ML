import h5py

# Open the HDF5 file in read mode
with h5py.File('autoencoder_L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausOcc_ae_080124.h5', 'r') as file:
    # Print the keys at the root level of the HDF5 file
    print("Root keys:", list(file.keys()))
