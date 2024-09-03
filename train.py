# U-Net Dataset Initialization, Training and Validation Set Iterations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# Use GPU if available
model = UNet()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = model.to(device)

# Training the 3D U-Net
class SpiderDataset(Dataset):
    
    '''
    Spider dataset class to load 3D images and corresponding masks.
    
    Attributes:
    -----------
    mask_path: list of str
        The list of paths to the corresponding masks of the input images.
    
    image_path: list of str
        The list of paths to the input images.
        
    transform : callable, optional
        A transformation to apply to the input image data, if data augmentation
        is needed.

    target_transform : callable, optional
        A transformation to apply to the input mask data, if data augmentation
        is needed.
        
    Methods:
    --------
    __len__():
        Returns the total number of samples in the dataset.
        
    __getitem__(idx):
        Fetches the image and mask with the corresponding index, creates the
        numpy arrays out of them and applies transformations (if defined).
        
    '''
    
    def __init__(self, mask_path, image_path, transform=None, target_transform=None):
        
        '''
        Initializes the SpiderDataset using paths to the masks and images.

        Parameters
        ----------
        mask_path: list of str
            The list of paths to the corresponding masks of the input images.
        
        image_path: list of str
            The list of paths to the input images.
            
        transform : callable, optional
            A transformation to apply to the input image data, if data augmentation
            is needed.

        target_transform : callable, optional
            A transformation to apply to the input mask data, if data augmentation
            is needed.

        Returns
        -------
        None.

        '''
        self.mask_path = mask_path
        self.image_path = image_path
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        
        '''
        Calculates the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples

        '''
        return len(self.image_path)
        
    def __getitem__(self, idx):
        
        '''
        Fetches the image and mask with the corresponding index, creates the
        numpy arrays out of them and applies transformations (if defined).

        Parameters
        ----------
        idx : int
            Index at the current iteration which defines the image and mask pair
            that will be fetched.

        Returns
        -------
        image_arr : numpy.ndarray
            The image as a numpy array.

        mask_arr : numpy.ndarray
            The mask as a numpy array.

        '''
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
    This function is used to get the data from SpiderDataset class and splits it
    into train, validation and test sets with determined densities for each set.

    Parameters
    ----------
    mrfiles_in : list of str
        List of names of MR image files that are sorted in pre-processing steps
        
    mr_volume_dir : str
        Path to the folder containing images
        
    mr_masks_dir : str
        Path to the folder containing masks

    Returns
    -------
    train_dataset : torch.utils.data.Subset
        Subset of image-mask pairs assigned to training subset.
        
    val_dataset : torch.utils.data.Subset
        Subset of image-mask pairs assigned to validation subset.
        
    test_dataset : torch.utils.data.Subset
        Subset of image-mask pairs assigned to test subset.

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

def remap_labels(mask, num_classes):
    
    '''
    Remaps the labels in the masks between 0 and 20 (at most).

    Parameters
    ----------
    mask : numpy.array
        Mask array.
        
    num_classes : int
        Number of classes in the current mask.

    Returns
    -------
    remapped_mask : numpy.array
        Mask array with a new, remapped labels.

    '''
    unique_labels = torch.unique(mask)
    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}

    # Apply remapping
    remapped_mask = mask.clone()
    for old_label, new_label in label_map.items():
        remapped_mask[mask == old_label] = new_label

    return remapped_mask



def unet_trainandvalidate(num_epochs, train_loader, val_loader):
    
    '''
    Trains and validates the U-Net model with specified hyperparameters and data 
    loaders by iterating over epochs and calculating loss, accuracy, and F1 scores
    for training and validation.

    Parameters
    ----------
    num_epochs: int
        Number of epochs specified as one of the hyperparameters.
        
    train_loader: torch.utils.data.dataloader.DataLoader
        Training data loader.
        
    val_loader: torch.utils.data.dataloader.DataLoader
        Validation data loader.

    Returns
    -------
    None.

    '''
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training step
    #Loss initialization
    t_loss, v_loss = [], [] #Initializing an empty list to store all training and validation losses for each epoch
    val_loss_min = float('inf')
    
    #Accuracy initialization
    t_acc, v_acc = [], []

    #F1 score initialization
    t_f1, v_f1 = [], []
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
            torch.save(model.state_dict(), best_model_path)
            val_loss_min = epoch_loss_val
            print(f'Saving model at epoch {epoch+1} with validation loss {epoch_loss_val:.4f}')
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {accuracy_train:.4f}, Val Accuracy: {accuracy_val:.4f}")
        
        
    # Store the loss, accuracy and F1 scores
    metrics_store = StoreMetrics(num_epochs)
    #Loss function
    metrics_store.store_loss(t_loss, v_loss, loss_out_path)
    #Accuracy function
    metrics_store.store_acc(t_acc, v_acc, acc_out_path)
    #F1 Score
    metrics_store.store_f1(t_f1, v_f1, f1_out_path)
    
    # Plot loss and accuracy scores
    MetricPlotter.plot_loss(t_loss, v_loss)
    MetricPlotter.plot_acc(t_acc, v_acc)