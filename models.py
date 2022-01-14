import torch
import math
import pdb

class LSTMUidModel(torch.nn.Module):
    def __init__(self, vocab_size=30522):
        super(LSTMUidModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=100, num_layers=2)
        self.dense = torch.nn.Linear(100, vocab_size)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, embedding):
        packed_output, (hidden_states, cell_states) = self.lstm(embedding)
        #unpacked_output, unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        dense_output = self.dense(hidden_states[-1])
        output = self.softmax(dense_output)
        return output

class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LSTMForSingleSequenceClassification(torch.nn.Module):
    def __init__(self, task, num_labels=2):
        super(LSTMForSingleSequenceClassification, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=100, num_layers=2)
        self.dense = torch.nn.Linear(100, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        self.task = task
    def forward(self, embedding):
        packed_output, (hidden_states, cell_states) = self.lstm(embedding)
        # unpacked_output, unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        dense_output = self.dense(hidden_states[-1])
        if self.task != 'stsb':
            output = self.softmax(dense_output)
        else:
            output = dense_output
        return output

class LSTMForBinarySequenceClassification(torch.nn.Module):
    def __init__(self, task, num_labels=2):
        super(LSTMForBinarySequenceClassification, self).__init__()
        if task == 'stsb':
            linear_input_size = 400
        else:
            linear_input_size = 200
        self.lstm1 = torch.nn.LSTM(input_size=300, hidden_size=100, num_layers=2)
        self.lstm2 = torch.nn.LSTM(input_size=300, hidden_size=100, num_layers=2)
        self.dense = torch.nn.Linear(linear_input_size, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.task = task
    def forward(self, embedding1, embedding2):
        packed_output1, (hidden_states1, cell_states1) = self.lstm1(embedding1)
        packed_output2, (hidden_states2, cell_states2) = self.lstm2(embedding2)
        if self.task != 'stsb':
            dense_output = self.dense(torch.cat((hidden_states1[-1], hidden_states2[-1]), dim=1))
            output = self.softmax(dense_output)
        else:
            dense_output = self.dense(torch.cat((hidden_states1[-1], hidden_states1[-2], hidden_states1[-1]*hidden_states2[-1], torch.abs(hidden_states1[-1]-hidden_states2[-1])), dim=1))
            #dense_output = self.dense(torch.cat((hidden_states1[-1], hidden_states2[-1]), dim=1))
            #pdb.set_trace()
            output = self.sigmoid(dense_output)
            #output = self.softmax(dense_output)
        return output

class LSTMForLanguageModelling(torch.nn.Module):
    def __init__(self, vocab_size=14741):
        super(LSTMForLanguageModelling, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=100, num_layers=2)
        self.dense = torch.nn.Linear(100, vocab_size)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, embedding):
        packed_output, (hidden_states, cell_states) = self.lstm(embedding)
        dense_output = self.dense(hidden_states[-1])
        output = self.softmax(dense_output)
        return output

class LSTMForSequenceTagging(torch.nn.Module):
    def __init__(self, n_class, n_hidden=100):
        super(LSTMForSequenceTagging, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=n_hidden, num_layers=6)
        self.fc = torch.nn.Linear(n_hidden, n_class)

    def forward(self, x):
        x, _ = self.lstm(x)
        #x_unpack, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #x = x.data.view(-1, x.data.shape[2])
        x = self.fc(x.data)
        return x


class LSTMForMultipleChoice(torch.nn.Module):
    def __init__(self, num_labels=2, input_size=300, hidden_size=100, num_layer=2):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layer)
        #  SentenceLength * (BatchSize * Choice) * InputSize -> Layer * (BatchSize * Choice) * HiddenSize
        self.lin = torch.nn.Linear(hidden_size, 1)
        #  BatchSize * Choice * HiddenSize -> BatchSize * Choice * 1
        self.num_labels = num_labels
        self.hidden_size = hidden_size

    def forward(self, x):
        output, (tmp, cn) = self.rnn(x)
        tmp = self.lin(tmp[-1].view(int(x.batch_sizes[0]/self.num_labels), self.num_labels, self.hidden_size))
        tmp = torch.softmax(tmp.view(tmp.shape[0], tmp.shape[1]), 1)
        return tmp
