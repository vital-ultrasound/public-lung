
import json
import os
import torch
import numpy as np
import luscp_utils.classifiers as classifiers
from PIL import Image

classes = ["A-lines", "B-lines", "Confluent B-line", "Consolidation", "Pleural Effussion"]

verbose = False
net = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = []


def get_classes():
    return classes


def initialize(input_size, model_path, verb: bool = False):
    """
    Load the model and initialize the network
    model_path: path where the config file and the model weights are.

    """
    global device
    global net
    global verbose
    global params

    config_filename = '{}/VITAL-lusclassification_training_config.json'.format(model_path)
    if os.path.exists(config_filename):
        with open(config_filename, "r") as lf:
            params = json.load(lf)
    else:
        print('Cannot find config file {}'.format(config_filename), flush=True)
        exit(-1)

    verbose = verb
    # load model for video classification
    print('[LUSclassificationp_worker.py: initialize] load model {}...'.format(model_path))

    n_output_classes = 5

    if params['model'] == 'tmpAttLSTM':
        net = classifiers.TempAttVideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'stAttLSTM':
        net = classifiers.SpatioTempAttVideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], spatial_attention_layers_pos=params['spatial_att_layers'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'LSTM':
        net = classifiers.VideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'tConv':
        net = classifiers.VideoClassifierConv(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])

    net.to(device)
    net.eval()

    checkpoint_f = '{}/best_validation_acc_model.pth'.format(model_path)

    if verbose:
        print('[LUS_classificationp_worker.py::initialize() - Load model {}'.format(checkpoint_f))
    state = torch.load(checkpoint_f)
    net.load_state_dict(state['model_state_dict'])

    if verbose:
        print(net)
    return True


def dowork(frames: np.array, verbose=0):
    with torch.no_grad():
        # pre-process the frames. Crop / resize in C++
        #frames = frames.transpose() # maybe do in cpp?
        #im = Image.fromarray(image_cpp)
        #im.save("/home/ag09/data/VITAL/input.png")
        frames = torch.from_numpy(frames).type(torch.float).to(device).unsqueeze(0).unsqueeze(0)/255.0

        # print(frames.shape)

        try:
            out = net(frames)
            if 'stAtt' in params['model']:
                sAtt = out[2]
                sAtt = [((sAtt_i - torch.min(sAtt_i)) / (torch.max(sAtt_i) - torch.min(sAtt_i)) * 255.0).type(torch.uint8).cpu().numpy() for sAtt_i in sAtt]
            if 'Att' in params['model']:
                att = out[1]
                att = (att - torch.min(att)) / (torch.max(att) - torch.min(att))
            out = out[0]
            out_index = torch.argmax(out, dim=1)

        except Exception as ex:
            print('[Python exception caught] LUSp::process() - {}{}'.format(ex, ex.__traceback__.tb_lineno))
            exit(-1)

    out = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
    # print('results')
    # print(out)
    # print(out_index)

    # average of all attentions
    attention = np.mean(np.stack(sAtt, axis=-1), axis=-1)
    # at at level att_idx
    #att_idx = 0
    #attention = sAtt[att_idx]
    #attention = frames[0,...].cpu().numpy()*255.0
    #print(attention.shape)
    #print(sAtt[att_idx].shape)

    #attention = attention[0, -1, ...] # take just the last frame
    # instead of taking the last frame, take a weighted average where the last takes the most weight
    var = 5
    weights = np.exp(np.expand_dims(np.array(np.arange(0, attention.shape[1]))/var,(0,2,3)))
    weights /= np.sum(weights)
    attention_weighted = np.sum(attention * weights, axis=1).squeeze() # take just the last frame

    #attention = np.ascontiguousarray(attention.transpose()).astype(np.uint8)
    attention_weighted = np.ascontiguousarray(attention_weighted).astype(np.uint8)
    #im = Image.fromarray(attention)
    #im.save("/home/ag09/data/VITAL/np_image.png")
    #np.save('/home/ag09/data/attention.npy', attention)
    #exit(-1)
    #print(attention.shape)
    return (out, attention_weighted.astype(np.uint8), attention)
