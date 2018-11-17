import matplotlib.pyplot as plt

def fig_plot(results, export_plot=False):
    # Plot results
    x = range(len(results['Train error']))
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

    # fig.subplots_adjust(top=0.80,bottom=0.2)
    fig.get_axes()[0].annotate(results['Name'], (0.5, 0.95),
                               xycoords='figure fraction', ha='center',
                               fontsize=16
                               )
    # fig.suptitle(results['Name'])
    # fig.tight_layout(pad=0.25)

    if export_plot:
        fig.savefig('./results/' + results['Name'])

    return fig
    # plt.imsave('1.png', fig)

    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].plot(range(num_epochs), train_loss_log, label='Training-set')
    # ax[0].plot(range(num_epochs), validation_loss_log, label='Validation-set')
    # ax[0].set_ylabel('Loss')
    # ax[0].legend(loc='best')
    # ax[0].grid(axis='both', which='both')
    # ax[0].set_title('Loss vs. Epochs')
    # ax[1].plot(range(num_epochs), train_error_log, label='Training-set')
    # ax[1].plot(range(num_epochs), validation_error_log, label='Validation-set')
    # ax[1].set_ylabel('Error [%]')
    # ax[1].set_xlabel('Epoch')
    # ax[1].legend(loc='best')
    # ax[1].grid(axis='both', which='both')
    # ax[1].set_title('Error vs. Epochs')
    #
    # # fig.subplots_adjust(top=0.80,bottom=0.2)
    # fig.suptitle(model.name)
    # fig.tight_layout(pad=0.25)
    #
    # # plt.imsave('1.png', fig)