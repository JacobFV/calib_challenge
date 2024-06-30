import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Adjust the import paths for custom modules
from commavq.utils.vqvae import Encoder, CompressorConfig


class Predictor(nn.Module):
    def __init__(self, encoded_dim, output_dim):
        super(Predictor, self).__init__()
        self.attention_pool = nn.AdaptiveAvgPool2d(1)  # Global attention pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        self.fc = nn.Linear(encoded_dim * 2, output_dim)  # Downprojection layer

    def forward(self, x):
        attention_out = self.attention_pool(x)  # Apply global attention pooling
        max_out = self.max_pool(x)  # Apply global max pooling
        merged = torch.cat((attention_out, max_out), dim=1)  # Merge the outputs
        merged = torch.flatten(merged, start_dim=1)  # Flatten the tensor
        return self.fc(merged)


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
predictor = Predictor(encoded_dim=config.z_channels, output_dim=2).to(DEVICE_NAME)

# Example training loop (simplified)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

for epoch in range(10):  # number of epochs
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE_NAME)
        targets = targets.to(DEVICE_NAME)
        with torch.no_grad():
            encoded_inputs = encoder(inputs)
        predictions = predictor(encoded_inputs)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
