import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import streamlit as st
import zipfile
import sys
import requests
import kagglehub

# Declare global variables
global data_set_path
global device
global loaded_model
global test_dataset
global user_test_transform

# Downloading the dataset
def download_dataset():
    global data_set_path  # Make sure to use the global variable
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Path to dataset files:", path)
    
    data_set_path = path
    Prepare_model(path)  # Ensure model is prepared after dataset download

def make_UI():
    label_to_openFile = 'Upload the X Ray Image'
    uploaded_file = st.file_uploader("Choose an X-Ray image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width =True)
        # st.write("File path:", uploaded_file.name)
        
        if st.button("Predict the tumor"):
            Predict_Tumer_v2(uploaded_file)

    else:
        st.write("Please upload an Brain X-Ray file.")

def Predict_Tumer_v2(user_image):
    st.write("Predicting the tumor")
    # st.write(f"Dataset path: {data_set_path}")  # Display dataset path
    
    # Convert the uploaded image to a tensor using the transformation pipeline
    image = Image.open(user_image).convert('RGB')  # Open and convert image to RGB
    image = user_test_transform(image).unsqueeze(0).to(device)  # Apply transformation and add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = loaded_model(image)  # Pass the image through the model
        _, predicted = torch.max(output.data, 1)  # Get the predicted class

    predicted_label = test_dataset.classes[predicted.item()]  # Map the predicted index to class name
    st.write(f"Predicted Tumer : {predicted_label}")

# Extract dataset from ZIP file
def extract_dataset(path):
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall("BrainTumerDataSet")

def Prepare_model(dataset_path):
    global device, loaded_model, test_dataset,user_test_transform  # Declare as global to use in Predict_Tumer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    class BrainTumorDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
            self.image_paths = []
            self.labels = []

            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    model_path = 'trained_model.pth'
    state_dict = torch.load(model_path, map_location=device)

    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(64 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 4)  # 4 classes

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = x.view(-1, 64 * 28 * 28)  # Flatten
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNNModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    test_dataset = BrainTumorDataset(root_dir=os.path.join(dataset_path, 'Testing'), transform=test_transform)
    loaded_model = model  # Assign model for use in predictions
    user_test_transform = test_transform
    # st.write("Model is ready for prediction")

if __name__ == "__main__":
    st.title("Brain Tumer Dectction Model")

    download_dataset()
    make_UI()
