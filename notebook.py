import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Adjust the import paths for custom modules
from commavq.utils.vqvae import Encoder, CompressorConfig


# Define a simple two-layer MLP with integrated Encoder
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, encoder):
        super(MLP, self).__init__()
        self.encoder = encoder  # Pretrained encoder
        self.layer1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.relu = nn.ReLU()
        self.layer2_res = nn.Linear(128, 128)  # Residual stream
        self.layer2_relu = nn.Linear(128, 128)  # ReLU stream
        self.merge = nn.Linear(256, 128)  # Merge layer
        self.downproject = nn.Linear(128, output_dim)  # Downproject layer

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            # x = x.view(x.size(0), -1)  # Flatten the encoder output
            x = x.float() / config.vocab_size  # Normalize encoder output
        x = self.layer1(x)
        x_res = self.layer2_res(x)
        x_relu = self.relu(self.layer2_relu(x))
        x_merged = torch.cat((x_res, x_relu), dim=1)
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
config = CompressorConfig()
encoder = Encoder(config)
encoder.load_state_dict_from_url(
    "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin",
    assign=True,
)
encoder = encoder.eval().to(device=DEVICE_NAME)

# Initialize the Predictor with the correct dimensions and integrated encoder
predictor = MLP(input_dim=196, output_dim=2, encoder=encoder).to(DEVICE_NAME)

# Example training loop (simplified)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

for epoch in range(10):  # number of epochs
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE_NAME)
        targets = targets.to(DEVICE_NAME)
        predictions = predictor(inputs)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(predictor.state_dict(), "predictor_model.pth")

# Load the model for testing
predictor_model = MLP(input_dim=196, output_dim=2, encoder=encoder).to(DEVICE_NAME)
predictor_model.load_state_dict(torch.load("predictor_model.pth"))
predictor_model.eval()  # Set the model to evaluation mode

# Testing loop
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE_NAME)
        targets = targets.to(DEVICE_NAME)
        predictions = predictor_model(inputs)
        loss = criterion(predictions, targets)
        print(f"Test Loss: {loss.item()}")
