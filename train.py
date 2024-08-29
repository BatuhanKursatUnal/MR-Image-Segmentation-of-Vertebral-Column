# U-Net Dataset Initialization, Training and Validation Set Iterations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

torch.cuda.empty_cache()

# Training the 3D U-Net
class SpiderDataset(Dataset):
    def __init__(self, mask_path, image_path, transform=None, target_transform=None):
        self.mask_path = mask_path
        self.image_path = image_path
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
       return len(self.image_path)
        
    def __getitem__(self, idx):
        mask_path = self.mask_path[idx]
        image_path = self.image_path[idx]
        mask = sitk.ReadImage(mask_path)        
        image = sitk.ReadImage(image_path)
        mask_arr = sitk.GetArrayFromImage(mask)
        image_arr = sitk.GetArrayFromImage(image)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image_arr, mask_arr
    
    
def get_dataset(mrfiles_in, mr_volume_dir, mr_masks_dir):
    '''
    This function is used to get the data from SpiderDataset class and split it into train, validation and test sets

    Parameters
    ----------
    mrfiles_in : MR image file names that are sorted in pre-processing steps (string)
    mr_volume_dir : Path to the folder containing images (string)
    mr_masks_dir : Path to the folder containing masks (string)

    Returns
    -------
    train_dataset : Image arrays in train set
    val_dataset : Image arrays in validation set
    test_dataset : Image arrays in test set

    '''
    images_list = []
    masks_list = []
    for files in mrfiles_in:
        mr_images_path = os.path.join(f"{mr_volume_dir}/rs_{files}")
        images_list.append(mr_images_path)
        mr_masks_path = os.path.join(f"{mr_masks_dir}/rs_mask_{files}")
        masks_list.append(mr_masks_path)
        
    spiderset = SpiderDataset(masks_list, images_list)
    
    train_size = int(len(spiderset) * 0.7) - (int(len(spiderset) * 0.7) % 4)
    val_size = int(len(spiderset) * 0.15) - (int(len(spiderset) * 0.15) % 4)
    test_size = ((len(spiderset) - train_size) - int(len(spiderset) * 0.15)) - ((len(spiderset) - train_size) - (int(len(spiderset) * 0.15) % 4))
    test_size += len(spiderset) - (train_size + val_size + test_size)

    densities = [train_size, val_size, test_size]
    
    dataset = torch.utils.data.random_split(spiderset, densities)

    train_dataset = dataset[0]
    val_dataset = dataset[1]
    test_dataset = dataset[2]
    
    return train_dataset, val_dataset, test_dataset
    

# Set the hyperparameters
learning_rate = 0.001
num_epochs = 16
batch_initial = 4
batch_later = 4

train_dataset, val_dataset, test_dataset = get_dataset(sorted_mr_imagefiles_new, images_volume_dir, masks_dir)

train_loader = DataLoader(
    dataset= train_dataset, 
    batch_size= batch_initial, 
    shuffle= True, 
    num_workers= 4,
    pin_memory= False,
    drop_last = False
    )

val_loader = DataLoader(
    dataset= val_dataset, 
    batch_size= batch_initial, 
    shuffle= False, 
    num_workers= 4,
    pin_memory= False,
    drop_last = False
    )

test_loader = DataLoader(
    dataset= test_dataset, 
    batch_size= batch_later, 
    shuffle= False, 
    num_workers= 4,
    pin_memory= False,
    drop_last = False
    )


model = UNet()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = model.to(device)
        
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


# Training step
#Loss initialization
t_loss = [] #Initializing an empty list to store all training losses for each epoch
v_loss = [] #Initializing an empty list to store all validation losses for each epoch
val_loss_min = float('inf')

#Accuracy initialization
t_acc = []
v_acc = []

#F1 score initialization
t_f1 = []
v_f1 = [] 
for epoch in range(num_epochs):
    model.train()
    batch_count_train = 0
    epoch_loss_train = 0
    epoch_accuracy_train = 0
    epoch_f1_train = 0
    for image_arr, mask_arr in train_loader:
        batch_count_train += 1
        
        image_arr, mask_arr = image_arr.to(device), mask_arr.to(device)
        image_arr = image_arr.unsqueeze(1)  #Adds a channel dimension at index 1
        mask_arr = remap_labels(mask_arr, num_classes=20)
        
        train_images = image_arr.to(device, dtype=torch.float32)
        train_masks = mask_arr.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
            
        #Feedforward
        outputs = model(train_images.float())
        t_preds = torch.argmax(outputs, dim=1)
        t_preds = t_preds.cpu().detach().numpy().flatten()
        t_masks = train_masks.cpu().detach().numpy().flatten()
        loss = criterion(outputs, train_masks)
        
        #Feedback and optimization
        loss.backward()
        optimizer.step()
        
        #Performance metrics
        #Loss
        epoch_loss_train += loss.item()
        #Accuracy
        epoch_accuracy_train += accuracy_score(t_masks, t_preds)
        #F1 score
        epoch_f1_train += f1_score(t_masks, t_preds, average = 'macro')
        
    epoch_loss_train /= batch_count_train
    epoch_accuracy_train /= batch_count_train
    epoch_f1_train /= batch_count_train

    print("\nTraining complete for this epoch.")
    t_loss.append(epoch_loss_train)
    t_acc.append(epoch_accuracy_train)
    t_f1.append(epoch_f1_train)


    # Validation step
    model = model.eval()
    
    with torch.no_grad():
        batch_count_val = 0
        epoch_loss_val = 0
        epoch_accuracy_val = 0
        epoch_f1_val = 0
        for image_arr, mask_arr in val_loader:
            batch_count_val += 1
            image_arr = image_arr.unsqueeze(1)  #Adds a channel dimension at index 1
            mask_arr = remap_labels(mask_arr, num_classes=20)
            
            val_images = image_arr.to(device, dtype=torch.float32)
            val_masks = mask_arr.to(device, dtype=torch.long)

            #Feedforward
            val_outputs =  model(val_images.float())
            v_preds = torch.argmax(val_outputs, dim=1)
            v_preds = v_preds.cpu().detach().numpy().flatten()
            v_masks = val_masks.cpu().detach().numpy().flatten()
            val_loss = criterion(val_outputs, val_masks)
            
            #Performance metrics
            #Loss
            epoch_loss_val += val_loss.item()
            #Accuracy
            epoch_accuracy_val += accuracy_score(v_masks, v_preds)
            #F1 score
            epoch_f1_val += f1_score(v_masks, v_preds, average = 'macro')
        
        epoch_loss_val /= batch_count_val
        epoch_accuracy_val /= batch_count_val
        epoch_f1_val /= batch_count_val
                
    v_loss.append(epoch_loss_val)
    v_acc.append(epoch_accuracy_val)
    v_f1.append(epoch_f1_val)
    
    # Pick out the best model
    if epoch_loss_val < val_loss_min:
        torch.save(model.state_dict(), '/Path/to/bestmodel.pth')
        val_loss_min = epoch_loss_val
        print(f'Saving model at epoch {epoch+1} with validation loss {epoch_loss_val:.4f}')
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {accuracy_train:.4f}, Val Accuracy: {accuracy_val:.4f}")


# Storing the losses
loss_data = pd.DataFrame({
'Epoch': list(range(1, num_epochs + 1)),
'Training Loss': t_loss,
'Validation Loss': v_loss
})
loss_data.to_csv('/Path/to/loss_data.csv', index=False)

# Storing the accuracy scores
acc_data = pd.DataFrame({
'Epoch': list(range(1, num_epochs + 1)),
'Training Accuracy': t_acc,
'Validation Accuracy': v_acc
})
acc_data.to_csv('/Path/to/accuracy.csv', index=False)

# Storing the F1 scores
f1_data = pd.DataFrame({
'Epoch': list(range(1, num_epochs + 1)),
'Training F1 Score': t_f1,
'Validation F1 Score': v_f1
})
f1_data.to_csv('/Path/to/f1.csv', index=False)


# Training and validation loss vs. epoch plot
plt.plot(t_loss, color='green')
plt.plot(v_loss, color='blue')
plt.title("Loss Calculation")
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

# Training and validation accuracy vs. epoch plot
plt.plot(t_acc, color='green')
plt.plot(v_acc, color='blue')
plt.title("Accuracy Calculation")
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.show()