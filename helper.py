from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Define the network architecture for our particular problem
def set_classifier(model, hidden_units):
    # Freeze parameters to avoid backprop
    for param in model.parameters():
        param.requires_grad = False

    input = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier

    return model

# Defining the training and validation loops
def train(epochs, print_every, trainloader, validloader, model, optimizer, criterion):
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        # training the network on training set
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                # running validation on validation set
                model.eval()
                with torch.no_grad():
                    for valid_input, labels in validloader:
                        valid_input, labels = valid_input.to(device), labels.to(device)
                        logps = model.forward(valid_input)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # calculating accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f'Epoch {epoch+1}/{epochs}',
                      'Training loss', running_loss/len(trainloader),
                      'Validation loss', test_loss/len(validloader),
                      'Accuracy', accuracy/len(validloader))

                running_loss = 0
                model.train()
