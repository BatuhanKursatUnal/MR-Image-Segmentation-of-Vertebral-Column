# Defining the U-Net class
import torch
import torch.nn as nn

class UNet(nn.Module):
    
    '''
    U-Net model for 3D image segmentation.

    This implementation is based on the 3D U-Net architecture, which consists of an 
    encoder-decoder structure together with skip connections. It is designed to
    handle 3D medical images.

    Attributes:
    -----------
    layer1, layer2, layer3, layer4, layer5: nn.Sequential
        The encoding layers of the network that extract features from the input image.
    
    adjust1, adjust2, adjust3, adjust4: nn.Conv3d
        3D convolutional layers used for adjusting the number of channels so 
        that the dimensions match in skip connections.
    
    upconv1, upconv2, upconv3, upconv4: nn.ConvTranspose3d
        Transpose convolutional layers used for up-sampling in the decoding process.
    
    decode_conv1, decode_conv2, decode_conv3, decode_conv4: nn.Sequential
        Convolutional layers for decoding.
    
    upconvfinal: nn.Conv3d
        A final 1x1 convolutional layer to output the final segmentation map.
        
    Methods:
    --------
    forward(x):
        Defines the forward pass of the U-Net model, including encoding, and decoding
        with skip connections.
        
    '''
    
    def __init__(self):
        
        '''
        Constracts the UNet model.

        Returns
        -------
        None.

        '''
        super().__init__()
        
        #Convolutional part of the encoder (3x3 convolutional matrix)
        def conv_relu(input_chans, output_chans):
            
            '''
            Convolutional part of the econding section.

            Parameters
            ----------
            input_chans : int
                Input channel dimension
                
            output_chans : int
                Output channel dimension

            Returns
            -------
            nn.Sequential
                Sequential applications of 3D convolutional layer followed by a
                3D batch normalization and a ReLU activation function, performed
                twice according to the original architecture of U-Net model.

            '''
            return nn.Sequential(
                nn.Conv3d(input_chans, output_chans, kernel_size=3, padding=1),
                nn.BatchNorm3d(output_chans),
                nn.ReLU(inplace=True),
                nn.Conv3d(output_chans, output_chans, kernel_size=3, padding=1),
                nn.BatchNorm3d(output_chans),
                nn.ReLU(inplace=True)
            )
    
        #Encoding layers
        self.layer1 = nn.Sequential(
            conv_relu(1, 32)
            )
    
        self.layer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            conv_relu(32, 64)
            )
    
        self.layer3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            conv_relu(64, 128)
            )
    
        self.layer4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            conv_relu(128, 256)
            )
        
        self.layer5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            conv_relu(256, 512)
            )
        
        #Adjusting number of channels for skip connections through 3D, 1x1 convolutional matrix
        self.adjust1 = nn.Conv3d(256, 512, kernel_size=1)
        self.adjust2 = nn.Conv3d(128, 256, kernel_size=1)
        self.adjust3 = nn.Conv3d(64, 128, kernel_size=1)
        self.adjust4 = nn.Conv3d(32, 64, kernel_size=1)
    
        #Upper convolutional part of the decoding part (2x2 convolutional matrix)
        self.upconv1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        
        
        #Decoder
        self.decode_conv1 = conv_relu(1024, 256)
        self.decode_conv2 = conv_relu(512, 128)
        self.decode_conv3 = conv_relu(256, 64)
        self.decode_conv4 = conv_relu(128, 32)
        
        self.upconvfinal = nn.Conv3d(32, 20, kernel_size=1) #1x1 convolutional matrix
        
    
    #Forward propagation
    def forward(self, x):
        
        '''
        Defines the forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            A 5D tensor of shape (batch_size, num_channels, depth, height, width)
            representing the input volume.

        Returns
        -------
        final_out : torch.Tensor
            The output of the network, a tensor of shape
            (batch_size, num_classes, depth, height, width).

        '''
        encode1 = self.layer1(x)
        encode2 = self.layer2(encode1)
        encode3 = self.layer3(encode2)
        encode4 = self.layer4(encode3)
        bottleneck = self.layer5(encode4)        
        
        #Decoder
        decode1 = self.upconv1(bottleneck)
        encode4 = self.adjust1(encode4)
        decode1 = torch.cat((decode1, encode4), dim=1) #Skip connection
        decode1 = self.decode_conv1(decode1)
        
        decode2 = self.upconv2(decode1)
        encode3 = self.adjust2(encode3)
        decode2 = torch.cat((decode2, encode3), dim=1) #Skip connection
        decode2 = self.decode_conv2(decode2)
        
        decode3 = self.upconv3(decode2)
        encode2 = self.adjust3(encode2)
        decode3 = torch.cat((decode3, encode2), dim=1) #Skip connection
        decode3 = self.decode_conv3(decode3)
        
        decode4 = self.upconv4(decode3)
        encode1 = self.adjust4(encode1)
        decode4 = torch.cat((decode4, encode1), dim=1) #Skip connection
        decode4 = self.decode_conv4(decode4)
        
        final_out = self.upconvfinal(decode4)
        
        return final_out