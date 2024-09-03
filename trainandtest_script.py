# U-Net training, validation and testing script
import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from train import get_dataset, unet_trainandvalidate
from test_unet import unet_test
import configparser

# Use GPU if available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

config = configparser.ConfigParser()
config.read('configuration.txt') #It must be stored in the same folder as the script

# Importing the paths and settings from configuration.txt file
best_model_path = config['paths']['best_model_path']
learning_rate = config['settings']['learning_rate']
num_epochs = config['settings']['num_epochs']
batch_initial = config['settings']['batch_initial']
batch_later = config['settings']['batch_later']
loss_out_path = config['settings']['loss_out_path']
acc_out_path = config['settings']['acc_out_path']
f1_out_path = config['settings']['f1_out_path']
mr_outputfolder_path = config['paths']['mr_outputfolder_path']
masks_outputfolder_path = config['paths']['masks_outputfolder_path']

model = UNet()
model = model.to(device)
torch.cuda.empty_cache()

train_dataset, val_dataset, test_dataset = get_dataset(sorted_mr_imagefiles_new, mr_outputfolder_path, masks_outputfolder_path)

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


unet_trainandvalidate(num_epochs, train_loader, val_loader)
unet_test(best_model_path, num_epochs, test_loader)


