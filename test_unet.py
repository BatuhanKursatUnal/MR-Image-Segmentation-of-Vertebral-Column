def visualize_results(images, masks, preds, slice_idx=None):
    """
    Visualize input images, ground truth masks, and predicted masks.

    Parameters
    ----------
    images: Batch of input images.
    masks: Batch of ground truth masks.
    preds: Batch of predicted masks.
    slice_idx: The slice index to visualize from the 3D volume.
               If None, the middle slice is chosen.
    """
    num_images = len(images)

    for i in range(num_images):
        plt.figure(figsize=(15, 5))

        # Determine the slice to visualize in case it is not given
        if slice_idx is None:
            slice_idx = images[i].shape[-1] // 2

        # Input Image
        plt.subplot(1, 3, 1)
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:
            img = img.squeeze(0)
        plt.imshow(img[..., slice_idx], cmap='gray')
        plt.title('Input Image (Slice {})'.format(slice_idx))

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        mask = masks[i].cpu().numpy()
        plt.imshow(mask[..., slice_idx], cmap='gray')
        plt.title('Ground Truth Mask (Slice {})'.format(slice_idx))

        # Predicted Mask
        plt.subplot(1, 3, 3)
        pred = preds[i].cpu().numpy()
        plt.imshow(pred[..., slice_idx], cmap='gray')
        plt.title('Predicted Mask (Slice {})'.format(slice_idx))

        plt.show()
        

# Testing set
#Loading the best model chosen in validation set
model = UNet()
model = model.to(device)
best_model = torch.load('/Path/to/bestmodel5.pth')
model.load_state_dict(best_model)

t2_loss = []
for epoch in range(num_epochs):
    model.eval()
    with torch.no_grad():
        epoch_loss_test = 0
        for image_arr, mask_arr in test_loader:
            image_arr = image_arr.unsqueeze(1)  #Adds a channel dimension at index 1
            mask_arr = remap_labels(mask_arr, num_classes=20)

            test_images = image_arr.to(device, dtype=torch.float32)
            test_masks = mask_arr.to(device, dtype=torch.long)

            test_outputs =  model(test_images.float())
            _, preds = torch.max(test_outputs, dim=1)
            test_loss = criterion(test_outputs, test_masks)

            visualize_results(test_images, test_masks, preds)

            #Validation loss in the current epoch
            epoch_loss_test += test_loss.item()/len(test_loader)

    print("\nTest complete for this epoch.")
    t2_loss.append(epoch_loss_test)

