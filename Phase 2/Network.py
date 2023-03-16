import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getLoss(pred, labels):
    criterion = nn.MSELoss()
    loss = criterion(pred, labels.view(-1, 8))
    return loss

class ModelBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = getLoss(out, labels)
        return loss
    

    
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = getLoss(out, labels)
        return {'val_loss': loss.detach()}
    
    def validation_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))










class NeRF(ModelBase):

    def __init__(self, embedding_dim_pos=4, embedding_dim_direction=8, hidden_dim=128):
        super(NeRF, self).__init__()

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction

        self.block1 = self.make_block(embedding_dim_pos * 6 + 3, hidden_dim, hidden_dim)
        self.block2 = self.make_block(hidden_dim + embedding_dim_pos * 6 + 3, hidden_dim, hidden_dim + 1)
        self.block3 = self.make_block(hidden_dim + embedding_dim_direction * 6 + 3, hidden_dim // 2, 3)

        self.relu = nn.ReLU()

    @staticmethod
    def make_block(in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)

        h = self.block1(emb_x)

        h = torch.cat((h, emb_x), dim=1)
        h = self.block2(h)
        sigma = self.relu(h[:, -1])
        h = h[:, :-1]

        h = torch.cat((h, emb_d), dim=1)
        h = self.block3(h)

        c = torch.sigmoid(h)

        return c, sigma
