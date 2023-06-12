import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights_normal2(m):
    if type(m) == nn.Linear:
        #torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight)
    elif type(m) == nn.Conv1d or type(m) == nn.Conv2d or type(m) == nn.Conv3d :
        #torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.normal_(m.weight)

def init_weights_normal(m):
    if type(m) == nn.Linear:
        #torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)
    elif type(m) == nn.Conv1d or type(m) == nn.Conv2d or type(m) == nn.Conv3d :
        #torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)

class extractlastcell(torch.nn.Module):

    def forward(self, x):
        out, _ = x
        return out #data for the last timestamp


class VideoClassifierLSTM(nn.Module):

    def __init__(self, input_size, n_output_classes, cnn_channels=(1, 16, 32), dropout_p=0.2, n_c_layers=2):
        """
        Video classifier without temporal attention implementing the model defined in:
            Kerdegari H, Nhat PT, McBride A, Razavi R, Van Hao N, Thwaites L, Yacoub S, Gomez A.
            "Automatic Detection of B-lines in Lung Ultrasound Videos from Severe Dengue Patients."
            In IEEE ISBI 2021 Apr 13 (pp. 989-993). IEEE.

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super().__init__()

        self.name = 'VideoClassifierLSTM'
        self.input_size = input_size
        self.n_output_classes = n_output_classes
        self.dropout_p = dropout_p

        self.n_c_layers = n_c_layers

        #self.dropout_layer_cnn = nn.Dropout(p=self.dropout_p / 2.0)
        #self.dropout_layer_dense = nn.Dropout(p=self.dropout_p)

        # define the CNN
        self.n_output_channels = cnn_channels
        self.kernel_size = (3, ) * (len(cnn_channels) -1)

        # define the flattener
        self.n_flattened_features_out = 256 # as in H's paper

        # define the lstm
        self.lstm_hidden_units = 16 # as in H's paper
        #self.lstm_dropout = dropout

        # compute the size at the output of each conv layer
        self.output_sizes = [self.input_size]
        maxpool_k, maxpool_stride = 3, 2 # kernel size and stride in maxpool



        #Implement CNN using 3D CNNS
        conv_padding, maxpool_padding = [], []
        for i in range(0, len(self.kernel_size)):
            conv_padding_i = math.floor(self.kernel_size[i] / 2)
            conv_padding.append(conv_padding_i)
            maxpool_padding_i = math.floor(maxpool_k / 2)
            maxpool_padding.append(maxpool_padding_i)

            h_out, w_out = self.output_sizes[-1][0] + 2 * conv_padding_i - (self.kernel_size[i] - 1), self.output_sizes[-1][
                1] + 2 * conv_padding_i - (self.kernel_size[i] - 1)  # after first conv
            h_out, w_out = h_out + 2 * conv_padding_i - (self.kernel_size[i] - 1), w_out + 2 * conv_padding_i - (
                        self.kernel_size[i] - 1)  # after second conv
            h_out, w_out = math.floor(
                (h_out + 2 * maxpool_padding_i - (maxpool_k - 1) - 1) / maxpool_stride + 1), math.floor(
                (w_out + 2 * maxpool_padding_i - (maxpool_k - 1) - 1) / maxpool_stride + 1)  # after max pool
            self.output_sizes.append((h_out, w_out))

        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels=self.n_output_channels[i], out_channels=self.n_output_channels[i + 1],
                          kernel_size=(1, self.kernel_size[i], self.kernel_size[i]), stride=1,
                          padding=(0, conv_padding[i], conv_padding[i])),
                nn.BatchNorm3d(self.n_output_channels[i + 1]),
                nn.ReLU(),
                #
                nn.Conv3d(in_channels=self.n_output_channels[i + 1], out_channels=self.n_output_channels[i + 1],
                          kernel_size=(1, self.kernel_size[i], self.kernel_size[i]), stride=1,
                          padding=(0, conv_padding[i], conv_padding[i])),
                nn.BatchNorm3d(self.n_output_channels[i + 1]),
                nn.ReLU(),
                #
                nn.MaxPool3d(kernel_size=(1, maxpool_k, maxpool_k), stride=(1, maxpool_stride, maxpool_stride),
                             padding=(0, maxpool_padding[i], maxpool_padding[i])),
            ) for i in range(0, len(self.kernel_size))])

        n_flattened_features_in = np.prod(self.output_sizes[-1]) * self.n_output_channels[-1] # to be calculated
        self.flattener = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_flattened_features_in, out_features=self.n_flattened_features_out),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        )

        #  define the LSTM module
        self.lstm_module = nn.LSTM(input_size=self.n_flattened_features_out,
                    batch_first=True, hidden_size=self.lstm_hidden_units,
                    bidirectional=True, # dropout=self.dropout_p, # removing dropout because it is just one layer and won't be applied
                    num_layers=1, bias=True)
        self.lstm = nn.Sequential(
            self.lstm_module,
            extractlastcell(), # needed because lstm returns a tuple and we only want the first element of the tuple
            nn.Dropout(p=self.dropout_p),
            #nn.Tanh(), # in hamideh's implementation she uses tanh inside (not after) LSTM. This is default in pytorch, so not neede dhere
        )

        # define the classification module
        n_classification_input_features = 2 * self.lstm_hidden_units
        classification_features = [ int(i * (self.n_output_classes-n_classification_input_features) / self.n_c_layers + n_classification_input_features) for i in range(self.n_c_layers+1) ]

        c_layers = []
        for i in range(self.n_c_layers):
            if i< (self.n_c_layers-1):
                extra_do  = nn.Dropout(p=self.dropout_p)
                extra_nla = nn.ReLU()
            else:
                extra_do, extra_nla = nn.Identity(), nn.Identity()

            layer = nn.Sequential(
                nn.Linear(in_features=classification_features[i], out_features=classification_features[i+1]),
                extra_nla,
                extra_do,
            )
            c_layers.append(layer)

        self.classification = nn.Sequential(*c_layers)

    def extra_repr(self):
        out_string = 'input_size={}'.format(self.input_size)
        return out_string

    def forward_spatial(self, data):

        l_i = data
        l_all = []
        for i in range(len(self.n_output_channels)-1):
            l_i = self.cnn_layers[i](l_i)
            l_all.append(l_i)

        cnn_out = l_i
        # flatten each time step now
        n_frames = data.shape[2]
        frame_features = []
        for i in range(n_frames):
            frame_i = cnn_out[:, :, i, ...]
            y = self.flattener(frame_i)
            frame_features.append(y)
        frame_features = torch.stack(frame_features, dim=1)  # output shape: batch size x nframes x 256

        return frame_features

    def temporal_integration(self, z):
        z_av = torch.mean(z, dim=1)
        return z_av

    def forward(self, data):
        frame_features = self.forward_spatial(data)
        # pass frame_features to the LSTM
        z = self.lstm(frame_features)
        z_out = self.temporal_integration(z)
        out = self.classification(z_out)
        return out

    def get_name(self):
        #linear_feat_str = '_features{}'.format(self.linear_features).replace(', ', '_').replace('(', '').replace(')', '')
        return self.name #+ linear_feat_str


class TemporalAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(TemporalAttentionBlock, self).__init__()
        #self.normalize_attn = normalize_attn
        self.W = nn.Parameter(torch.tensor(torch.randn(in_features, 1), requires_grad=True))
        self.bias = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.use_bias = True
        #self.compatibility_score = nn.Conv1d(in_channels=in_features, out_channels=1,
        #                                     kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        logits = x @ self.W # multiply
        if self.use_bias:
            logits += self.bias

        M, _ = torch.max(logits, dim=1, keepdim=True) # maximum for each sample in the batch
        ai = torch.exp(logits - M)
        eps = torch.finfo(torch.float32).eps #
        att_weights = ai / (torch.sum(ai, axis=1, keepdims=True) + eps) # sum over all frames

        weighted_input = x * att_weights

        # now sum over the number of frames
        weighted_pooled_input = torch.sum(weighted_input, axis=1)
        return att_weights, weighted_pooled_input

class TempAttVideoClassifierLSTM(VideoClassifierLSTM):

    def __init__(self, input_size, n_output_classes, cnn_channels, dropout_p, n_c_layers):
        """
        Video classifier with temporal attention implementing the model defined in:
            Kerdegari H, Nhat PT, McBride A, Razavi R, Van Hao N, Thwaites L, Yacoub S, Gomez A.
            "Automatic Detection of B-lines in Lung Ultrasound Videos from Severe Dengue Patients."
            In IEEE ISBI 2021 Apr 13 (pp. 989-993). IEEE.

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super().__init__(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=cnn_channels, dropout_p=dropout_p, n_c_layers=n_c_layers)
        self.name = 'TempAttVideoClassifierLSTM'

        self.attention_layer_classification = TemporalAttentionBlock(in_features=self.lstm_hidden_units * 2, normalize_attn=True)

    def forward(self, data):
        frame_features = self.forward_spatial(data)

        # pass frame_features to the LSTM
        z = self.lstm(frame_features)

        # seq. weighted attention
        #p = self.projector_classification(z)
        attention, g_i = self.attention_layer_classification(z)  #
        #z_av = torch.mean(z, dim=1)
        out = self.classification(g_i)
        return out, attention

class MultiFrameLinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(MultiFrameLinearAttentionBlock, self).__init__()
        self.in_features = in_features
        self.normalize_attn = normalize_attn
        self.compatibility_score = nn.Conv3d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        """
        Implementation inspired from https://github.com/SaoYan/LearnToPayAttention/blob/master/blocks.py
        l : feature map
        g : reference
        """
        N, C, Nf, W, H = l.size() # B= batch size, C = n. channels. N = n. frames
        # N, C, W, H = l.size()
        c = self.compatibility_score(l+g) # batch_sizexOut_channelsxNxWxH

        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, Nf, -1), dim=3).view(N, 1, Nf, W, H)
        else:
            a = torch.sigmoid(c)


        #c_exp  =torch.exp(c)
        #c_exp_sum = torch.sum(torch.sum(torch.exp(c), dim=3, keepdim=True), dim=4, keepdim=True)  # batch_sizexOut_channelsxNx1x1
        #attention_coef = (c_exp / c_exp_sum) # batch_sizexOut_channelsxWxH , also force to be positive and between 0 and 1


        # now, multiply and dotprod, then sum over space (global average pooling)
        #g = torch.mul(attention_coef.expand_as(l), l).view(N, C, Nf, -1).sum(dim=-1)
        g = torch.mul(a.expand_as(l), l)

        if self.normalize_attn:
            g = g.view(N, C, Nf, -1).sum(dim=3)  # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C, Nf)

        a = c.view(N, 1, Nf, W, H)

        # this function returns two tensors:
        #   - attention_coef of size # batch_size x N x W x H
        #   - g of size # batch_size x in_features x n_frames

        #return attention_coef.squeeze(1), g
        return a.squeeze(1), g

class SpatioTempAttVideoClassifierLSTM(TempAttVideoClassifierLSTM):

    def __init__(self, input_size, n_output_classes, cnn_channels, dropout_p, n_c_layers, spatial_attention_layers_pos=(1,)):
        """
        Video classifier with spatial and temporal attention implementing the model defined in:
            Kerdegari H, Nhat PT, McBride A, Razavi R, Van Hao N, Thwaites L, Yacoub S, Gomez A.
            "Automatic Detection of B-lines in Lung Ultrasound Videos from Severe Dengue Patients."
            In IEEE ISBI 2021 Apr 13 (pp. 989-993). IEEE.

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super().__init__(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=cnn_channels, dropout_p=dropout_p, n_c_layers=n_c_layers)
        self.name = 'SpatioTempAttVideoClassifierLSTM'

        in_channels = self.n_output_channels[-1]
        self.spatial_attention_layers_pos = spatial_attention_layers_pos # attention after cnn layer 1 and 2

        """
        Attention layer for a time resolved feature. One for each layer
        """
        self.spatial_attention_layers = nn.ModuleList(
            [MultiFrameLinearAttentionBlock(in_features=self.n_output_channels[-1], normalize_attn=True) for i in self.spatial_attention_layers_pos])

        """ 
        get to the same number of channels as the last layer
        """
        self.projectors_layers = nn.ModuleList(
            [nn.Conv3d(in_channels=self.n_output_channels[i + 1], out_channels=self.n_output_channels[-1], kernel_size=1, padding=0,
                       bias=False) for i in self.spatial_attention_layers_pos])

        """
        self.dense: do a conv with a kernel of the same size as the last layer, to do a kind of global pooling
        """
        self.dense = nn.Sequential(
            nn.Conv3d(in_channels=self.n_output_channels[-1], out_channels=self.n_output_channels[-1],
                      kernel_size=(1, self.output_sizes[-1][0], self.output_sizes[-1][1]) , padding=0),
            nn.ReLU(True),
            nn.BatchNorm3d(self.n_output_channels[-1]),
        )

        # assuming we put the attention on the last layer
        self.n_flattened_features_out = len(self.spatial_attention_layers_pos) * self.n_output_channels[-1]

        self.lstm_module = nn.LSTM(input_size=self.n_flattened_features_out,
                                   batch_first=True, hidden_size=self.lstm_hidden_units,
                                   bidirectional=True, #dropout=self.lstm_dropout, # removing dropout because it is just one layer and won't be applied
                                   num_layers=1, bias=True)
        self.lstm = nn.Sequential(
            self.lstm_module,
            extractlastcell(),  # needed because lstm returns a tuple and we only want the first element of the tuple
            nn.Dropout(p=self.dropout_p),
            # nn.Tanh(), # in hamideh's implementation she uses tanh inside (not after) LSTM. This is default in pytorch, so not neede dhere
        )

        self.upsampler = nn.Upsample(size=tuple(self.input_size), mode='bilinear', align_corners=True)


    def forward_spatial_attention(self, data):

        l_i = data
        l_all = []
        for i in range(len(self.n_output_channels)-1):
            l_i = self.cnn_layers[i](l_i)
            l_all.append(l_i)

        g = self.dense(l_i)

        # attentions
        spatial_attentions, gs = [], []
        for i, layer_pos in enumerate(self.spatial_attention_layers_pos):
            p = self.projectors_layers[i](l_all[layer_pos])
            a, g_i = self.spatial_attention_layers[i](p, g)  #
            att = self.upsampler(a)

            spatial_attentions.append(att)
            gs.append(torch.permute(g_i, (0, 2, 1) ))

        if len(gs) > 0:
            g = torch.cat(gs, dim=2)  # batch_sizexNxC
        #g.squeeze_(dim=2) # just in case there is just one attention layer


        return g, spatial_attentions

    def forward(self, data):
        frame_features, spatial_attentions = self.forward_spatial_attention(data)

        # pass frame_features to the LSTM
        z = self.lstm(frame_features)

        # seq. weighted attention
        #p = self.projector_classification(z)
        temporal_attention, g_i = self.attention_layer_classification(z)  #
        #z_av = torch.mean(z, dim=1)
        out = self.classification(g_i)

        return out, temporal_attention, spatial_attentions


class VideoClassifierConv(VideoClassifierLSTM):

    def __init__(self, input_size, n_output_classes, cnn_channels, dropout_p, n_c_layers):
        """
        Video classifier implementing the model defined in:
            Kerdegari H, Nhat PT, McBride A, Razavi R, Van Hao N, Thwaites L, Yacoub S, Gomez A.
            "Automatic Detection of B-lines in Lung Ultrasound Videos from Severe Dengue Patients."
            In IEEE ISBI 2021 Apr 13 (pp. 989-993). IEEE.
            but using temporal conv instead of LSTM

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super().__init__(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=cnn_channels, dropout_p=dropout_p, n_c_layers=n_c_layers)
        self.name = 'VideoClassifierConv'

        self.kernel_size_t = 3
        self.n_output_channels_t = (64, 2 * self.lstm_hidden_units) # to match the LSTM
        self.maxpool_k_t = 3
        self.maxpool_stride_t = 2

        self.tConv1d = nn.Sequential(
            nn.Conv1d(in_channels=self.n_flattened_features_out, out_channels=self.n_output_channels_t[0],
                      kernel_size=self.kernel_size_t, stride=1, padding = int((self.kernel_size_t-1)/2)),
            nn.Dropout(p=self.dropout_p_cnn),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_output_channels_t[0]),
            nn.MaxPool1d(kernel_size=self.maxpool_k_t, stride=self.maxpool_stride_t, padding = int((self.maxpool_k_t-1)/2)),
            nn.Conv1d(in_channels=self.n_output_channels_t[0], out_channels=self.n_output_channels_t[1],
                      kernel_size=self.kernel_size_t, stride=1, padding = int((self.kernel_size_t-1)/2)),
            nn.Dropout(p=self.dropout_p_cnn),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_output_channels_t[1]),
            nn.MaxPool1d(kernel_size=self.maxpool_k_t, stride=self.maxpool_stride_t, padding = int((self.maxpool_k_t-1)/2)),
        )


    def tConv(self, frame_features):
        # frame_features is BS x ntime frames x n features
        # Let's turn features into channels to apply a 1D conv
        frame_features_permuted = torch.permute(frame_features, (0, 2, 1))
        out = self.tConv1d(frame_features_permuted)
        # undo the permute
        out_unpermuted = torch.permute(out, (0, 2, 1))
        return out_unpermuted

    def forward(self, data):
        frame_features = self.forward_spatial(data)

        # pass frame_features to the LSTM
        #z = self.lstm(frame_features)
        z = self.tConv(frame_features)
        z_out = self.temporal_integration(z)
        out = self.classification(z_out)
        return out