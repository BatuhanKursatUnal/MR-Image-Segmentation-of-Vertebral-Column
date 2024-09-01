# Storing and plotting the metrics

class StoreMetrics:
    
    '''
    A class for storing numerous metrics; loss, accuracy, and F1 score.
    
    Attributes:
    -----------
    num_epochs: int
        The total number of epochs in the training and validation loops
        
    Methods:
    --------
    store_loss(t_loss, v_loss, loss_out_path): 
        Method to store the loss data in a dataframe.
    
    store_acc(t_acc, v_acc, acc_out_path):
        Method to store the accuracy data in a dataframe.
        
    store_f1(t_f1, v_f1, f1_out_path):
        Method to store the F1 score data in a dataframe.
        
    
    '''
    
    def __init__(self, num_epochs):
        
        '''
        Initializes the StoreMetrics to store metrics such as loss, accuracy,
        and F1 score, both for training and validation sets.

        Parameters
        ----------
        num_epochs: int
            The total number of epochs in the training and validation loops

        Returns
        -------
        None.

        '''
        self.num_epochs = num_epochs
        
    def store_loss(self, t_loss, v_loss, loss_out_path):
        
        '''
        Method to store the loss data in a dataframe.
        
        It saves the dataframe into a csv file in the specified path.

        Parameters
        ----------
        t_loss: list
            List of training loss values for each epoch.
            
        v_loss: list
            List of validation loss values for each epoch.
            
        loss_out_path: str
            Output path to which the csv file will be saved.

        Returns
        -------
        None.

        '''
        loss_data = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Training Loss': t_loss,
        'Validation Loss': v_loss
        })
        loss_data.to_csv(loss_out_path, index=False)
        
    def store_acc(self, t_acc, v_acc, acc_out_path):
        
        '''
        Method to store the accuracy data in a dataframe.
        
        It saves the dataframe into a csv file in the specified path.

        Parameters
        ----------
        t_acc: list
            List of training accuracy values for each epoch.
            
        v_acc: list
            List of validation accuracy values for each epoch.
            
        acc_out_path: str
            Output path to which the csv file will be saved.

        Returns
        -------
        None.

        '''
        acc_data = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Training Accuracy': t_acc,
        'Validation Accuracy': v_acc
        })
        acc_data.to_csv(acc_out_path, index=False)
        
    def store_f1(self, t_f1, v_f1, f1_out_path):
        
        '''
        Method to store the F1 score data in a dataframe.
        
        It saves the dataframe into a csv file in the specified path.

        Parameters
        ----------
        t_f1: list
            List of training F1 scores for each epoch.
            
        v_f1: list
            List of validation F1 scores for each epoch.
            
        f1_out_path: str
            Output path to which the csv file will be saved.

        Returns
        -------
        None.

        '''
        f1_data = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Training Accuracy': t_f1,
        'Validation Accuracy': v_f1
        })
        f1_data.to_csv(f1_out_path, index=False)



# Plot loss and accuracy with respect to epoch number

class MetricPlotter:
    
    '''
    A class for plotting numerous loss and accuracy for training and validation sets.
    
        
    Methods:
    --------
    plot_loss(t_loss, v_loss): 
        Plots the loss function for training and validation.
    
    plot_acc(t_acc, v_acc):
        Plots the accuracy function for training and validation.
        
    
    '''
    
    @staticmethod
    def plot_loss(self, t_loss, v_loss):
        
        '''
        Plots the loss function for training and validation.

        Parameters
        ----------
        t_loss: list
            List of training loss values for each epoch.
            
        v_loss: list
            List of validation loss values for each epoch.

        Returns
        -------
        None.

        '''
        plt.plot(t_loss, color='green')
        plt.plot(v_loss, color='blue')
        plt.title("Loss Calculation")
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.legend(['Training loss', 'Validation loss'])
        plt.show()
        
    @staticmethod 
    def plot_acc(self, t_acc, v_acc):
        
        '''
        Plots the accuracy function for training and validation.

        Parameters
        ----------
        t_acc: list
            List of training accuracy values for each epoch.
            
        v_acc: list
            List of validation accuracy values for each epoch.

        Returns
        -------
        None.

        '''
        plt.plot(t_acc, color='green')
        plt.plot(v_acc, color='blue')
        plt.title("Accuracy Calculation")
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.legend(['Training accuracy', 'Validation accuracy'])
        plt.show()
