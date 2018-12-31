import os
import numpy as np
import matplotlib.pyplot as plt

path = r'D:\לימודים\97200 - Deep learning\HW1\results'
# models = [model for model in os.listdir(path) if model.endswith('.npy')]
#
# for m in models:
#     model = np.load(os.path.join(path, m)).tolist()
#     epoch, e_min = model['Validation error'].argmin()+1, model['Validation error'].min()
#     print('{}: {}, epoch: {}'.format(model['Name'], e_min, epoch))

file = r'Model 4 - Split image-16, lr=0.001, wd=0.0001, bs=64.npy'
results = np.load(os.path.join(path, file)).tolist()
x = range(1, 1+len(results['Train error']))
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10,10))
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
plt.show()