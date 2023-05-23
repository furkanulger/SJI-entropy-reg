import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, models, transforms
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score


torch.manual_seed(42) # same seed for the same weight initialization.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Model training arguments")
parser.add_argument("-t", "--train_test", required=True, help=r"'train' or 'test' the model")
parser.add_argument("-m", "--model", required=True, help="The model to be trained/tested: type 'googlenet' 'resnet50' 'resnet18' 'vgg16'")
parser.add_argument("-a", "--skew", help="Skewness parameter (alpha) value [0,1]", default=0.5)
parser.add_argument("-p", "--path", help="Path for the model to be tested")
parser.add_argument("-c", "--num_classes", help="Number of classes for the model", default=2)
parser.add_argument("-bs", "--batchsize",help="Batch size for the model", default=8)
parser.add_argument("-lr", "--lrnrate", help="Learning rate for the model", default=1e-4)
parser.add_argument("-e", "--numEpochs", help="Number of epochs for the model", default=150)
args = parser.parse_args() 
print(args)

save_PATH = './results/skewed_JSD_model_'
os.makedirs("./dataset/train", exist_ok=True)
os.makedirs("./dataset/validation", exist_ok=True)
os.makedirs("./dataset/test", exist_ok=True)
os.makedirs("./results", exist_ok=True)


if args.model == "googlenet":
    net = models.googlenet(pretrained=False).to(device)
    net.aux_logits = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, args.num_classes).to(device) #for 2 classes
elif args.model == "resnet50":
    net = models.resnet50(pretrained=False).to(device)  #
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, args.num_classes).to(device)
elif args.model == "resnet18":
    net = models.resnet18(pretrained=False).to(device)  #
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, args.num_classes).to(device)
elif args.model == "vgg16":
    net = models.vgg16(pretrained=False).to(device)
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, args.num_classes).to(device)


trainloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(r'dataset/train/', transforms.Compose([
        transforms.Resize((84, 84)),  # width, height
        transforms.ToTensor()
    ])), batch_size=args.batchsize, num_workers=0, shuffle=False)

validloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(r'dataset/valid/', transforms.Compose([
        transforms.Resize((84, 84)),  # width, height
        transforms.ToTensor()
    ])), batch_size=args.batchsize, num_workers=0, shuffle=False)

testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(r'dataset/test/', transforms.Compose([
        transforms.Resize((84, 84)),  # width, height
        transforms.ToTensor()
    ])), batch_size=1, num_workers=0, shuffle=False)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1 * torch.mean(b.sum(dim=1))
        return b


class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()
        self.epsilon = 0.00001

    # Calculation of KL divergence between probability distributions of discrete random variables. Dkl(P || Q) P: data dist., Q: approx. dist.
    def forward(self, x, labels):
        b = F.one_hot(labels, args.num_classes) * (F.log_softmax(x, dim=1)
                                              - torch.log(F.one_hot(labels, args.num_classes) + self.epsilon))
        b = -torch.mean(b.sum(dim=1))
        return b



class skewed_JSD():
    def __init__(self, a, bs, num_epochs, lrn_rate):
        self.a = a
        self.num_epochs = num_epochs
        self.lrn_rate = lrn_rate
        self.bs = bs
        self.beta = 1

    def train(self):
        print("Training the model with mini-batch size: {0}, learning rate: {1}, epoch: {2}, beta: {3}".format(self.bs,
                                                                                                               self.lrn_rate,
                                                                                                               self.num_epochs,
                                                                                                               self.beta))

        valid_acc_max = -np.Inf
        entropy = HLoss()
        CE_loss = nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lrn_rate, weight_decay=1e-3)

        for epoch in range(self.num_epochs):
            total_train_loss = 0
            total_JSdiv = 0
            total_cross_ent = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                ce_loss = CE_loss(outputs, labels)
                H_loss = entropy(outputs)

                a_JSdiv = 1 / (self.a * (1 - self.a)) * (-(1 - self.a) * np.log(2) - self.a * H_loss
                                                         + entropy((1 - self.a) * (1 / 2) + self.a * F.softmax(outputs, dim=1)))
                loss = ce_loss + self.beta * a_JSdiv

                loss.backward()
                optimizer.step()  # gradient descent. w' = w - n. grad(loss)/grad(w)
                total_JSdiv += a_JSdiv.item() * inputs.size(0)
                total_cross_ent += ce_loss.item() * inputs.size(0)
                total_train_loss += loss.item() * inputs.size(0) # loss over the mini-batch.

            # calculate average losses
            train_loss = total_train_loss / len(trainloader.sampler)
            cross_ent = total_cross_ent / len(trainloader.sampler)
            JSdiv = total_JSdiv / len(trainloader.sampler)
            print("Epoch: {0:d}, alpha-JS divergence: {1:.3f}, CE loss: {2:.3f}, train loss: {3:.3f}".format(epoch, JSdiv, cross_ent, train_loss))

            ######################
            # validate the model #
            ######################
            correct = 0
            total = 0
            net.eval()  # prep model for evaluation
            for img, labels in validloader:
                inputs = img.to(device)
                labels = labels.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = net(inputs)
                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

            valid_acc = correct / total
            print('Epoch: {:d} \tValidation accuracy: {:.4f}'.format(
                epoch, valid_acc))
            if valid_acc > valid_acc_max:
                print('Validation accuracy increased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                    valid_acc_max,
                    valid_acc))
                torch.save(net.state_dict(), save_PATH + "epoch" + str(epoch) + "_valAcc_" + str(round(valid_acc, 3)) + "_maxValidacc.pth")
                valid_acc_max = valid_acc


        torch.save(net.state_dict(), save_PATH + ".pth")
        print('Finished Training')

    def test_model(self, load_PATH):
        correct = 0
        total = 0
        predicted_lbls=[]
        groundtruth_labels=[]
        net.load_state_dict(torch.load(load_PATH))
        net.eval()

        entropy= HLoss()
        total_H = 0
        total_JSdiv = 0
        # 0: abnormal, 1:normal
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.to(device))

                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                H_loss = entropy(outputs)  # calculates the conditional entropy
                a_JSdiv = 1 / (self.a * (1 - self.a)) * (-(1 - self.a) * np.log(2) - self.a * H_loss
                                                         + entropy((1 - self.a) * (1 / 2) + self.a * F.softmax(outputs, dim=1)))

                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

                total_H += H_loss
                total_JSdiv += a_JSdiv
                # class 0: defective class 1: intact.
                predicted_lbls.append(predicted.cpu().detach().numpy())
                groundtruth_labels.append(labels.cpu().detach().numpy())


        recall= recall_score(y_true=groundtruth_labels, y_pred=predicted_lbls, average='weighted')
        precision= precision_score(y_true=groundtruth_labels, y_pred=predicted_lbls, average='weighted')
        F1_score = f1_score(groundtruth_labels, predicted_lbls)


        print("Accuracy of the network on the test images: {0:.3f} %".format(100 * (correct / total)))
        print("precision: {0:.5f}, recall= {1:.5f}, f1 score= {2:.5f}.".format(
            precision, recall, F1_score))
        print("Average entropy of the model: {0:.5f}.".format(
            total_H.detach().cpu().numpy() / testloader.__len__()))
        print("Average skew JSD of the model on the test set: {0:.5f}.".format(
            total_JSdiv.detach().cpu().numpy() / testloader.__len__()))


if args.train_test == "train":
    skewed_JSD(a=float(args.skew), bs= args.batchsize, num_epochs=args.numEpochs, lrn_rate= args.lrnrate).train()
else:
    skewed_JSD(a=float(args.skew), bs= args.batchsize, num_epochs=args.numEpochs, lrn_rate= args.lrnrate).test_model(args.path)
