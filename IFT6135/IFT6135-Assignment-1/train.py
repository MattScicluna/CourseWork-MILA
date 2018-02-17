import torch
from dataset import newsDataset
from torch.utils.data import DataLoader


def main(batch_size=64, num_hidden=100, max_epochs=500, learning_rate=1e-4):
    train_data = newsDataset(in_data='20news-bydate/matlab/train.data',
                             out_data='20news-bydate/matlab/train.label',
                             transform=None)

    dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    input_size, output_size = train_data.in_data[0].shape[0], 1

    #  Build the model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(num_hidden, output_size),
    )
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        for i_batch, (data, target) in enumerate(dataloader):
            pred = model(data)
            loss = loss_fn(pred, data)
            model.zero_grad()
            loss.backward()

            for param in model.parameters():
                param.data -= learning_rate * param.grad.data

            if i_batch % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i_batch * len(data), len(train_data),
                           100. * i_batch / len(train_data), loss.data[0]))

if __name__ == '__main__':
    main()
