import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedded = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers = num_layers, batch_first = True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=2048)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=2048, out_features=4096)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_features=4096, out_features=vocab_size)
        self.bias_init()
    
    def bias_init(self):
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
    
    def forward(self, features, captions):
        # feature shape: (batch size, embed size)
        # caption shape: (batch size, caption length)
        
        # get rid of <end>
        captions = captions[:, :-1] # caption shape: (batch size, caption length-1)
        # embed captions
        embedded = self.embedded(captions) # embeded shape: (batch size, caption length-1, embed size)
        # add features to the head of embed as the first input to LSTM
        features = features.unsqueeze(1) # match the shape of embedded features shape: (batch size, 1, embed size)
        embedded = torch.cat((features, embedded), dim=1) # embeded shape: (batch size, caption length, embed size)
        # initialize hidden state and cell state
        self.batch_size = embedded.shape[0] # get the batch size
        # lstm
        lstm_out, self.hidden = self.lstm(embedded)
        # fc layers
        fc_out = self.drop1(F.relu(self.fc1(lstm_out)))
        fc_out = self.drop2(F.relu(self.fc2(fc_out)))
        fc_out = F.log_softmax(self.fc3(fc_out), dim=2)
        return fc_out      

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for i in range(20):
            if states == None:
                lstm_out, states = self.lstm(inputs)
            else:
                lstm_out, states = self.lstm(inputs, states)
            fc_out = F.relu(self.fc1(lstm_out))
            fc_out = F.relu(self.fc2(fc_out))
            fc_out = F.log_softmax(self.fc3(fc_out), dim=2)
            _, output = torch.max(fc_out, dim=2)
            output_numpy = output.data.cpu().numpy()[0].item()
            caption.append(output_numpy)
            if output_numpy == 1:
                break
            inputs = self.embedded(output)
        return caption