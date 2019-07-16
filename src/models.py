# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def accuracy(pos_samples, neg_samples, device):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
    """
    margin = 0
    pred = (pos_samples - neg_samples + margin).cpu().data
    acc = (pred > 0).sum()*1.0 / pos_samples.size()[0]
    acc = torch.from_numpy(np.array([acc], np.float32))
    acc = acc.to(device)
    return Variable(acc)


def calc_padding_size(i, o, k, s=1, d=1):
    p = int(((o-1)*s + k + (k-1)*(d-1) - i) / 2)
    return p


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbeddingNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.convnet_1 = self._make_conv_net(4)
        self.convnet_2 = self._make_conv_net(8)
        self.convnet_3 = self._make_conv_net(16)
        self.convnet_4 = self._make_conv_net(32)

        p = calc_padding_size
        i0, i1 = input_size
        self.convnet = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=(1, 8), stride=(1, 8),
                padding=(p(i0, i0, 1, 1), p(i1, i1, 8, 2))
                ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=(8, 1), stride=(2, 1),
                padding=(p(i0, i0, 8, 2), p(i1, i1, 1, 1))
                ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.AvgPool2d(6, ceil_mode=True),
            )

        # global average pooling
        self.pooling = lambda x: F.avg_pool2d(x, kernel_size=x.size()[2:])


        #self.fc = nn.Sequential(
        #    nn.Dropout(0.5),
        #    nn.Linear(128, output_size),
        #    )

    def _make_conv_net(self, filter_size):
        p = calc_padding_size
        i0, i1 = self.input_size

        convnet = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=(1, filter_size), stride=(1, 2),
                padding=(p(i0, i0, 1, 1), p(i1, i1, filter_size, 2))
                ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32,
                kernel_size=(filter_size, 1), stride=(2, 1),
                padding=(p(i0, i0, filter_size, 2), p(i1, i1, 1, 1))
                ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=(1, filter_size), stride=(1, 2),
                padding=(p(i0, i0, 1, 1), p(i1, i1, filter_size, 2))
                ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=(filter_size, 1), stride=(2, 1),
                padding=(p(i0, i0, filter_size, 2), p(i1, i1, 1, 1))
                ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        return convnet

    def forward(self, x):
        embedding = self.convnet_1(x)
        embedding += self.convnet_2(x)
        embedding += self.convnet_3(x)
        embedding += self.convnet_4(x)
        embedding = self.convnet(embedding)
        embedding = self.pooling(embedding)
        embedding = embedding.view(embedding.size()[0], -1)
        #embedding = self.fc(embedding)
        #embedding /= embedding.pow(2).sum(1, keepdim=True).sqrt()  # normalize
        return embedding


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        embed_anc = self.embedding_net(anchor)
        embed_pos = self.embedding_net(positive)
        embed_neg = self.embedding_net(negative)
        return embed_anc, embed_pos, embed_neg

