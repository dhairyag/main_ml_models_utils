'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss():
    from main import train_losses, test_losses
    fig, axs = plt.subplots(1,2,figsize=(10,10))
    axs[0].plot(train_losses)
    axs[0].set_title("Training Loss")
    axs[1].plot(test_losses)
    axs[1].set_title("Test Loss")
    plt.show()

def plot_accuracy():
    from main import train_acc, test_acc
    fig, axs = plt.subplots(1,2,figsize=(10,10))
    axs[0].plot(train_acc)
    axs[0].set_title("Training Accuracy")
    axs[1].plot(test_acc)
    axs[1].set_title("Test Accuracy")
    plt.show()




def misclassified_10(model):
    # Set the model to evaluation mode
    model.eval()
    from main import testloader, device, class_names

    # List to store misclassified images and actual labels
    misclassified_images = []
    misclassified_labels = []
    actual_labels = []

    # Loop through the test dataset
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            incorrect_mask = predicted != labels
            misclassified_images.append(images[incorrect_mask])
            misclassified_labels.append(predicted[incorrect_mask])
            actual_labels.append(labels[incorrect_mask])

    # Concatenate the list of misclassified images
    misclassified_images = torch.cat(misclassified_images)
    misclassified_labels = torch.cat(misclassified_labels)
    actual_labels        = torch.cat(actual_labels)
    # Plot the misclassified images
    plt.figure(figsize=(13, 13))
    plt.suptitle("Misclassified Images", fontsize=14)
    
    dataset_mean = [0.49139968, 0.48215827 ,0.44653124] 
    dataset_std = [0.24703233, 0.24348505, 0.26158768]

    for i in range(10):
        plt.subplot(5, 2, i + 1)
        
        for j in range(misclassified_images[i].shape[0]):
            misclassified_images[i][j] = (misclassified_images[i][j]*dataset_std[j])+dataset_mean[j]
        
        plt.imshow(np.transpose(misclassified_images[i].cpu().numpy(), (1, 2, 0)))
        mis_lab = misclassified_labels[i].item()
        act_lab = actual_labels[i].item()
        plt.title(f"Predicted: {class_names[mis_lab]}({mis_lab}),\n Actual: {class_names[act_lab]}({act_lab})", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    # If terminal size cannot be determined, set a default width
    term_width = 40

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



