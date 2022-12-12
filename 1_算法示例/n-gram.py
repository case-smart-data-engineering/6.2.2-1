#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2  # the same as window_size
EMBEDDING_DIM = 10
test_sentence = "When forty winters shall besiege thy brow,And dig deep trenches in thy beauty's field,Thy youth's proud livery so gazed on now,Will be a totter'd weed of small worth held:Then being asked, where all thy beauty lies,Where all the treasure of thy lusty days;To say, within thine own deep sunken eyes,Were an all-eating shame, and thriftless praise.How much more praise deserv'd thy beauty's use,If thou couldst answer 'This fair child of mineShall sum my count, and make my old excuse,'Proving his beauty by succession thine!This were to be new made when thou art old,And see thy blood warm when thou feel'st it cold.".split()

vocb = set(test_sentence)  # remove repeated words
word2id = {word: i for i, word in enumerate(vocb)}
id2word = {word2id[word]: word for word in word2id}


# define model
class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        # super(NgramModel, self)._init_()
        super().__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        # the first step: transmit words and achieve word embedding. eg. transmit two words, and then achieve (2, 100)
        emb = self.embedding(x)
        # the second step: word wmbedding unfold to (1,200)
        emb = emb.view(1, -1)
        # the third step: transmit to linear model, and then use relu, at last, transmit to linear model again
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        # the output dim of last step is the number of words, wo can view as a classification problem
        # if we want to predict the max probability of the words, finally we need use log softmax
        log_prob = F.log_softmax(out)
        return log_prob


ngrammodel = NgramModel(len(word2id), CONTEXT_SIZE, 100)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngrammodel.parameters(), lr=1e-3)

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]

for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for data in trigram:
        # we use 'word' to represent the two words forward the predict word, we use 'label' to represent the predict word
        word, label = data  # attention
        word = Variable(torch.LongTensor([word2id[e] for e in word]))
        label = Variable(torch.LongTensor([word2id[label]]))
        # forward
        out = ngrammodel(word)
        loss = criterion(out, label)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss: {:.6f}'.format(running_loss / len(word2id)))

# predict
word, label = trigram[3]
word = Variable(torch.LongTensor([word2id[i] for i in word]))
out = ngrammodel(word)
_, predict_label = torch.max(out, 1)
predict_word = id2word[predict_label.item()]
print('real word is {}, predict word is {}'.format(label, predict_word))
