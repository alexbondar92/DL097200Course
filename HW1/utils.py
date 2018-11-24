import matplotlib.pyplot as plt
import torch

import models

input_size = 784
num_classes = 10

def fig_plot(results, export_plot=False):
    # Plot results
    x = range(1, 1+len(results['Train error']))
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, results['Train loss'], label='Training-set')
    ax[0].plot(x, results['Validation loss'], label='Validation-set')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')
    ax[0].grid(axis='both', which='both')
    ax[0].set_title('Loss vs. Epochs')
    ax[1].plot(x, results['Train error'], label='Training-set')
    ax[1].plot(x, results['Validation error'], label='Validation-set')
    ax[1].set_ylabel('Error [%]')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='best')
    ax[1].grid(axis='both', which='both')
    ax[1].set_title('Error vs. Epochs')

    fig.get_axes()[0].annotate(results['Name'], (0.5, 0.95),
                               xycoords='figure fraction', ha='center',
                               fontsize=16
                               )


    if export_plot:
        fig.savefig('./results/' + results['Name']+'.png')

    return fig

def save_checkpoint(model, optimizer, filepath):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filepath)

def load_checkpoint(model, optimizer, filepath):
    # "lambda" allows to load the model on cpu in case it is saved on gpu
    state = torch.load(filepath, lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    return model, optimizer

def initialize_model(model_num):
    if model_num == 1:
        return models.model1(input_size, num_classes)
    elif model_num == 2:
        return models.model2(input_size, num_classes)
    elif model_num == 3:
        return models.model3(input_size, num_classes)
    elif model_num == 4:
        return models.model4(input_size, num_classes)
    elif model_num == 5:
        return models.model5(input_size, num_classes)
    elif model_num == 6:
        return models.model6(input_size, num_classes)
    elif model_num == 7:
        return models.model7(input_size, num_classes)
