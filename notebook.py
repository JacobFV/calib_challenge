import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Adjust the import paths for custom modules
from commavq.utils.vqvae import Encoder, CompressorConfig


# Define a simple two-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.relu = nn.ReLU()
        self.exp = nn.ReLU()  # Exponential activation
        self.layer2_res = nn.Linear(128, 128)  # Residual stream
        self.layer2_relu = nn.Linear(128, 128)  # ReLU stream
        self.layer2_exp = nn.Linear(128, 128)  # Exponential stream
        self.merge = nn.Linear(
            384, 128
        )  # Merge layer, output now matches input of downproject layer
        self.downproject = nn.Linear(128, output_dim)  # Downproject layer

    def forward(self, x):
        x = self.layer1(x)
        x_res = self.layer2_res(x)
        x_relu = self.relu(self.layer2_relu(x))
        x_exp = self.exp(self.layer2_exp(x))
        x_merged = torch.cat(
            (x_res, x_relu, x_exp), dim=1
        )  # Concatenate along the feature dimension
        x_merged = self.merge(x_merged)
        x = self.downproject(x_merged)
        return x


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = frames.transpose((0, 3, 1, 2))  # Change from B H W C to B C H W
    return frames


def load_ds():
    # Update the path to the video and label files
    video_frames = extract_frames("labeled/0.hevc")
    angles = np.loadtxt("labeled/0.txt")
    assert len(video_frames) == len(angles), "Mismatch in frames and angles count!"
    X_train, X_test, y_train, y_test = train_test_split(
        video_frames, angles, test_size=0.2, random_state=42
    )
    train_tensor = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
    return train_loader, test_loader


# Load data
train_loader, test_loader = load_ds()

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

# Load and configure the encoder
# load model
config = CompressorConfig()
with torch.device("meta"):
    encoder = Encoder(config)
encoder.load_state_dict_from_url(
    "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin",
    assign=True,
)
encoder = encoder.eval().to(device=DEVICE_NAME)

# Now you can initialize the Predictor with the correct output dimension


# Initialize the Predictor with the correct dimensions
predictor = MLP(input_dim=196, output_dim=2).to(DEVICE_NAME)

# Example training loop (simplified)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

for epoch in range(10):  # number of epochs
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE_NAME)
        targets = targets.to(DEVICE_NAME)
        with torch.no_grad():
            encoded_inputs = encoder(inputs)
            encoded_inputs = encoded_inputs.float() / config.vocab_size
        predictions = predictor(encoded_inputs)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
