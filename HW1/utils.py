import matplotlib.pyplot as plt
import torch

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