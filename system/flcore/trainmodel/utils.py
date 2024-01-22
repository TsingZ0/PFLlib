from flcore.trainmodel.models import *
import torchvision
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

def get_model(args, vocab_size = 98635, max_len=200, emb_dim=32):
    model_str = args.model
    # Generate args.model
    if model_str == "mlr": # convex
        if "mnist" in args.dataset:
            return Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            return Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
        else:
            return Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

    elif model_str == "cnn": # non-convex
        if "mnist" in args.dataset:
            return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "omniglot" in args.dataset:
            return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        elif "Digit5" in args.dataset:
            return Digit5CNN().to(args.device)
        else:
            return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

    elif model_str == "dnn": # non-convex
        if "mnist" in args.dataset:
            return DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            return DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
        else:
            return DNN(60, 20, num_classes=args.num_classes).to(args.device)
    
    elif model_str == "resnet":
        return torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        
        # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
    
    elif model_str == "resnet10":
        return resnet10(num_classes=args.num_classes).to(args.device)
    
    elif model_str == "resnet34":
        return torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "alexnet":
        return alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        # args.model = alexnet(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        
    elif model_str == "googlenet":
        return torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)
        
        # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "mobilenet_v2":
        return mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        # args.model = mobilenet_v2(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        
    elif model_str == "lstm":
        return LSTMNet(hidden_dim=args.emb_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

    elif model_str == "bilstm":
        return BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.emb_dim, output_size=args.num_classes, 
                    num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                    embedding_length=emb_dim).to(args.device)

    elif model_str == "fastText":
        return fastText(hidden_dim=args.emb_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

    elif model_str == "TextCNN":
        return TextCNN(hidden_dim=args.emb_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                        num_classes=args.num_classes).to(args.device)

    elif model_str == "Transformer":
        return TransformerModel(ntoken=args.vocab_size, d_model=args.emb_dim, nhead=8, d_hid=emb_dim, nlayers=2, 
                        num_classes=args.num_classes).to(args.device)
    
    elif model_str == "AmazonMLP":
        return AmazonMLP().to(args.device)

    elif model_str == "harcnn":
        if args.dataset == 'har':
            return HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
        elif args.dataset == 'pamap':
            return HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)

    else:
        raise NotImplementedError
