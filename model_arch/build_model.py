from .model_resnet import ResNet18, ResNet34, ResNet50, ResNet20
# from resnets.model_resnet_official import ResNet50
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from .FastText import FastText



def build_model(args):

    if args.model == 'resnet18':
        netglob = ResNet18(args.num_classes)

    elif args.model == 'resnet20':
        netglob = ResNet20(args.num_classes)

    elif args.model == 'resnet34':
        netglob = ResNet34(args.num_classes)

    elif args.model == 'resnet50':
        netglob = ResNet50(args.num_classes)

    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, args.num_classes)
        
    elif args.model == 'fasttext':
        # FastText model arch
        vocab_size = args.vocab_size
        embedding_dim = args.hidden_dim
        netglob = FastText(embedding_dim, vocab_size=vocab_size, num_classes=args.num_classes)

    else:
        exit('Error: unrecognized model')

    return netglob
