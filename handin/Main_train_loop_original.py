#!/usr/bin/env python3
# -- File info --#
__author__ = 'Ida L. Olsen'
__contributors__ = ''
__contact__ = ['s174020@student.dtu.dk']
__version__ = '0'
__date__ = '2021-10-08'

# -- Built-in modules -- #
# import sys
# sys.path.append('/work3/s174020/')

# -- Third-part modules -- #
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import jaccard_score
# from UNET_corr import UNet

# -- Proprietary modules -- #
from R2_loss import r2_loss

def Train_Loop(model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloaders: torch.utils.data.Dataset,
                 EPOCHS: int = 100,
                 save_dir: str = '/work3/s174020/plots/for_show/',
                 LR: float = 10**(-3),
                 name = 'ADAM-augmented_aug2'
                 ):

    # computes number of training and validation steps per epoch
    train_steps = len(dataloaders['train'])
    val_steps = len(dataloaders['val'])
    # epoch_min = 0

    # define training hyperparameters
    EPOCHS = EPOCHS

    # initialize a dictionary to store training history
    H = {
          "train_loss": [],
          "val_R2_score": [],
          "val_loss": [],
          "val_acc": []
    }

    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    val_loss_min = 10000  # start with large number
    r2_score_max = -10  # start with a small number
    # val_loss_previous = torch.zeros(1).cuda()
    for EPOCH in range(0, EPOCHS):
        # from tqdm import trange
        batch = tqdm(dataloaders['train'])
        model.train()
        # initialize total training and validation loss
        train_loss = 0
        val_loss = 0

        # initialize number of correct predictions in training and validation step
        train_correct = 0
        val_correct = 0
        accuracy = []
        for inputs, labels, ID in batch:
            # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

            # perform the forward pass and compute the training loss
            inputs = inputs.cuda()
            y_pred = model(inputs)

            labels[labels > 12] = 0

            labels = labels.cuda()

            loss = criterion(y_pred, labels.long())

            print("Loss:", loss)

            # Zeros out the gradients of the optimizer, perform the backpropagation step
            # and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total training loss so far and
            # calculate number of correct predictions
            train_loss += loss
            # argmax returns largest number along dimension
            # (we want along the channel dimension to get the number predicted
            # with the highest certainty)
            # check where prediction are right and convert to binary
            # preds_corr = (y_pred.argmax(1) == labels).type(torch.float)
            # convert to binary
            # train_correct += preds_corr.sum().item()

            # idx = pred.argmax(axis=1).cpu().numpy()[0]
            # ll = labels.cpu().squeeze()
            
            
        # scheduler.step()
        """--------------------------------------------"""
        # Validation training loop
        """--------------------------------------------"""

        # switching off gradient tracking and computation
        with torch.no_grad():
            # set model in evaluation mode (switch off parts of model that
            # behave different during trainign and validation)
            model.eval()
            # loop over validation dataset
            batch_val = tqdm(dataloaders['val'])
            count = 0
            lab = torch.tensor(())
            predictions = torch.tensor(())
            for inputs, labels, ID, mask, index_nan_data, lat, lon in batch_val:
                count += 1
                inputs = inputs.cuda()
                # gives a matrix with dimension image_size X num classes.
                # Hence giving the likelihood for each class at each location
                y_pred = model(inputs)

                # information about parameters
                print('shape of labels:'+str(labels.shape)+'\n')
                print('shape of mask:'+str(mask.shape)+'\n')

                labels = labels.cuda()
                # Calculates loss based on validation data
                val_loss += criterion(y_pred, labels.long())

                # Label predictions - find the prediction giving max likelihood
                pred_label = y_pred.argmax(axis=1).cpu()[0]

                # base score on non zero class
                ll = labels.cpu().squeeze()

                # preds_corr = (y_pred.argmax(1) == labels.type(torch.float))
                # # convert to binary
                # val_correct += preds_corr.sum().item()

                # number of correct predictions - not including nans
                preds_corr = (pred_label[ll != 0].numpy() == ll[ll != 0].numpy())

                # number of correct pixels per validation image
                val_correct += preds_corr.sum().item() / len(ll[ll != 0])

                ## saving predictions and labels to compute R2 score
                lab = torch.cat((lab, ll[ll != 0].type(torch.double).flatten()), 0)
                predictions = torch.cat((predictions, pred_label[ll != 0].type(torch.double).flatten()), 0)




        # Statistical evaluation of training and validation

        # Calculate the average training and validation loss
        avg_training_loss = train_loss / train_steps
        avg_validation_loss = val_loss / val_steps
        r2_score = r2_loss(lab, predictions)

        # if val_loss < val_loss_min:
        #     val_loss_min = val_loss
        #     accuracy_best = val_correct
        #     epoch_min = EPOCH
        # find location of maximum r2 score
        if r2_score > r2_score_max:
            r2_score_max = r2_score
            accuracy_best = val_correct
            epoch_min = EPOCH

        # Calculate training and validation accuracy
        # (number of avergae samples per image that are correct)
        # train_correct = train_correct / (len(dataloaders['train'].dataset)*(256**2))
        val_correct = val_correct / (len(dataloaders['val'].dataset))
        # train_accuracy = np.meadian(accuracy)
        # update our training history
        H["train_loss"].append(avg_training_loss.cpu().detach().numpy())
        H["val_R2_score"].append(r2_score)
        H["val_loss"].append(avg_validation_loss.cpu().detach().numpy())
        H["val_acc"].append(val_correct)

        # Print model trainig and validation informaiton
        print("[INFO] EPOCH: {} / {}".format(EPOCH + 1, EPOCHS))  # starts from 0
        print("Testing loss: {:.4f}, testing accuracy: {:.4f}".format(avg_validation_loss, val_correct))

        # finish measuring how long training took
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))
        # Evaluate the network
        print("[INFO] evaluating network...")

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["val_loss"], label="test_loss")
        plt.plot(H["val_R2_score"], label="test_R2_score")
        if EPOCH > 5:
            # plt.scatter(epoch_min, (val_loss_min / val_steps).cpu().detach().numpy(), label="Minimum loss location")
            plt.scatter(epoch_min, (r2_score_max).cpu().detach().numpy(), label="Max R2 location")
        # plt.plot(H["train_acc_jacard"], label="train_acc_jacard")
        plt.plot(H["val_acc"], label="test_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.ylim([0, 3])
        if EPOCH % 30 == 0 and EPOCH > 1:
            plt.savefig(save_dir+name+str(EPOCH)+str(LR)+'_training_loop.png')
            plt.show()
        # serialize the model to disk
        model_name = "ASIP_model"+str(EPOCH)
        torch.save(model.state_dict(), model_name)

    return H, val_loss_min, accuracy_best / (len(dataloaders['val'].dataset)*256**2), epoch_min
