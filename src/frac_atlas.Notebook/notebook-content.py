# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "99d33961-ba47-4c2b-9ca5-46e478fe9e50",
# META       "default_lakehouse_name": "frac_atlas",
# META       "default_lakehouse_workspace_id": "f72f59d8-9015-4de8-bab1-573ef077c941"
# META     },
# META     "environment": {
# META       "environmentId": "f453af4c-bf78-4d80-9d8d-87b4f3cb17e9",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# CELL ********************

%pip install torch==2.4.0 torchvision

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%pip install --upgrade mlflow

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os
import io
import json
import numpy as np
import pandas as pd
import shutil
import random
import torch
import torchvision
import h5py
from PIL import Image, ImageFile
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.pytorch

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

torch.manual_seed(0)

print('Using PyTorch version', torch.__version__)

root_dir = '/fracatlas_pytorch/fracatlas'
write_dir = 'Files/output'
train_dir = '/lakehouse/default/Files/fracatlas/train'
test_dir = '/lakehouse/default/Files/fracatlas/test'
validation_dir = '/lakehouse/default/Files/fracatlas/val'
non_fractured_dir = '/lakehouse/default/Files/fracatlas/not fractured'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train_image_filepaths = [
    x for x in os.listdir(f"{train_dir}/img")
]
train_annotation_filepaths = [f"{train_dir}/ann/{x}" for x in os.listdir(f"{train_dir}/ann")]

test_image_filepaths = [x for x in os.listdir(f"{test_dir}/img")]
test_annotation_filepaths = [f"{test_dir}/ann/{x}" for x in os.listdir(f"{test_dir}/ann")]

print(f"Loaded {len(train_image_filepaths)} image files for training")
print(f"Loaded {len(test_image_filepaths)} image files for testing")



class_names = ["not fractured", "fractured"]
print(f"The two classnames observed are '{class_names[0]}' and '{class_names[1]}'")




train_image_filepaths.extend(test_image_filepaths)
fractured_image_filepaths = train_image_filepaths
fractured_annotation_filepaths = train_annotation_filepaths.extend(test_annotation_filepaths)

nonfractured_image_filepaths = [
    x for x in os.listdir(f"{non_fractured_dir}/img")
]
nonfractured_annotation_filepaths = [x for x in os.listdir(f"{non_fractured_dir}/ann")]

print(f"Loaded {len(fractured_image_filepaths)} image files of fractured bones")
print(f"Loaded {len(nonfractured_image_filepaths)} image files of non-fractured bones")


'''

for c in class_names:
    images = [x for x in os.listdir(os.path.join(write_dir, c))]
    selected_images = random.sample(images, int(.05*len(images)))
    
    for image in selected_images:
        source_path = os.path.join(write_dir, c, image)
        target_path = os.path.join(write_dir, 'test', c, image)
        shutil.move(source_path, target_path)
'''


class BoneXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, class_names, transform):
        self.images = {}
        self.transform = transform
        self.labels = []
        self.class_names = class_names
        self.image_dirs = image_paths
        
        def get_images(class_name):
            images = [
                x for x in os.listdir(image_paths[class_name])
            ]
            print(f'Found {len(images)} {class_name} examples')
            
            return images
        
        for c in self.class_names:
            self.images[c] = get_images(c)
        
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB') # 3 channels for resnet18
        
        return self.transform(image), self.class_names.index(class_name)



train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])


test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])


train_dirs = {
    'not fractured': f'{non_fractured_dir}/img',
    'fractured': f'{train_dir}/img'
}
print(train_dirs)
train_dataset = BoneXRayDataset(
    train_dirs, class_names, train_transform
)

test_dirs = {
    'not fractured': f'{non_fractured_dir}/img',
    'fractured': f'{test_dir}/img'
}
print(test_dirs)
test_dataset = BoneXRayDataset(
    test_dirs, class_names, test_transform
)


batch_size = 6

dl_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

dl_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print('Num of training batches', len(dl_train))
print('Num of test batches', len(dl_test))


def show_images(images, labels, preds):
    plt.figure(figsize=(8,4))
    
    for i, image in enumerate(images):
        plt.subplot(1,6, i + 1, xticks=[], yticks=[])
        
        image = image.numpy().transpose((1,2, 0)) # To adhere to matplotlib needs
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        
        col = 'green' if preds[i] == labels[i] else 'red'
        
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].cpu().numpy())]}', color=col)
        
    plt.tight_layout()
    plt.show()


images, labels = next(iter(dl_train))
show_images(images, labels, labels)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Setup Resnet50 Model
resnet = torchvision.models.resnet50(pretrained=True)
num_filters = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_filters, 1) # set up for binary classification
reset = resnet.to(device)
print(resnet) # print model structure



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# 
# # Setup VGG16 Model
# vgg16 = torchvision.models.vgg16(pretrained=True)
# num_ftrs = vgg16.classifier[6].in_features
# vgg16.classifier[6] = torch.nn.Linear(num_ftrs, 1) # set up for binary classification
# vgg16 = vgg16.to(device)
# 
# print(vgg16) # print model structure
# 
# # Setup DenseNet Model
# dense_net = torchvision.models.densenet121(pretrained=True)
# num_ftrs = dense_net.classifier.in_features
# dense_net.classifier = torch.nn.Linear(num_ftrs, 1)  # set up for binary classification
# dense_net = dense_net.to(device)
# 
# print(dense_net) # print model structure


# CELL ********************

def show_preds(model):
    model.eval()
    images, labels = next(iter(dl_test))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs.to(device), 1)
    show_images(images, labels, preds)


# Train Model
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for train_step, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs.to(device))
        targets = targets.view(-1, 1).float()
        
        loss = criterion(outputs.to(device), targets.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        preds = torch.sigmoid(outputs.to(device))
        preds = (preds > 0.5).float()
        
        correct += (preds == targets.to(device)).sum().item()
        total += targets.size(0)
        
        if train_step % 20 == 0:
            print('#', sep=' ', end='', flush=True)
    
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        mlflow.log_metric('train_loss', loss.data.item(), train_step)
        mlflow.log_metric('train_accuracy', epoch_accuracy, train_step)
    
    print(f'\n\nTraining Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    return epoch_loss, epoch_accuracy

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for eval_step, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs.to(device))
            targets = targets.view(-1, 1).float()  # Ensure targets are of shape [batch_size, 1]
            
            loss = criterion(outputs.to(device), targets.to(device))
            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            correct += (preds.to(device) == targets.to(device)).sum().item()
            total += targets.size(0)
            
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            if eval_step % 20 == 0:
                print(f'Validation Loss: {(running_loss / total):.4f}, Accuracy: {(correct / total):.4f}')

                show_preds(model)
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    print(f'\n\nValidation Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    return epoch_loss, epoch_accuracy ,accuracy, precision, recall, f1, roc_auc


# Run Model (Train/Evaluation Loop)
def run_model(model, dl_train, dl_test, model_name: str, best_accuracy: float):
    criterion = torch.nn.BCEWithLogitsLoss()
    total_epochs = 0
    
    # Learning Rate
    learning_rate = 3e-5

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while(True):
        print(f"\nCurrent Epoch: {total_epochs+1}")

        # TRAIN MODEL
        print("Training Model...")
        train_loss, train_accuracy = train_model(model, dl_train, criterion, optimizer)
        print("Model Training Complete")

        # EVALUATE MODEL
        print("Validating Model...")
        val_loss, val_accuracy, accuracy, precision, recall, f1, roc_auc = evaluate_model(
            model, dl_test, criterion
        )
        
        print("Model Validation Complete")

        print(f"\nEpoch {total_epochs+1} Summary")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        total_epochs += 1 # increment epochs

        # Save the best model
        if val_accuracy > best_accuracy:
            print(f'\n{model_name.upper()} Model performance condition met')
            
            best_accuracy = val_accuracy
            
            print(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )
            print(
                f"F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}"
            )
            break


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Set given experiment as the active experiment. If an experiment with this name does not exist, a new experiment with this name is created.
mlflow.set_experiment("frac_atlas")
mlflow.autolog(exclusive=False)

ImageFile.LOAD_TRUNCATED_IMAGES = True

mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False, # Update this property to enable custom logging
    disable_for_unsupported_versions=False,
    silent=False
)
mlflow.pytorch.autolog(registered_model_name='fracatlas')

with mlflow.start_run() as run:

    run_model(resnet, dl_train, dl_test, "resnet50", .85)

    #signature = mlflow.models.infer_signature(
    #    dl_train,
    #    resnet(dl_train).detach().numpy(),
    #)

    model_info = mlflow.pytorch.log_model(
        resnet, 
        'frac_atlas'
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

