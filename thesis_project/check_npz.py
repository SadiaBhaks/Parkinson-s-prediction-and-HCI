import numpy as np

data = np.load("val_data.npz")
print("Arrays inside:", data.files)
print("X shape:", data["X"].shape)
print("y shape:", data["y"].shape)
print("First 10 labels:", data["y"][:10])
print("First few values of participant 0's data:")
print(data["X"][0, :3, :5])   # first 3 timesteps, first 5 channels