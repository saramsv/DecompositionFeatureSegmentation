import argparse 
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from util import AverageMeter, load_model
import os
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from size_constrained_clustering import fcm, equal, minmax, shrinkage
# by default it is euclidean distance, but can select others
from sklearn.metrics.pairwise import haversine_distances




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/usb/sara_img/', help='path to the root dir that has a dir for images in it, and in that dir there are dirs for classes')
    parser.add_argument('--img_dir_names', default='img_dir_names', help='this is a file with name of the classes (dirs) in each line')
    parser.add_argument('--clustering', type=str, 
                        choices=['Kmeans', 'PIC'], default='Kmeans')
    parser.add_argument('--extension', type=str, default='jpg') 
    
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--batch', default=1, type=int,
                        help='mini-batch size (default: 128)')
    return parser.parse_args()

def compute_features(dataloader, model, N):
    batch_time = AverageMeter()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features


def main(args):
    seed = 31
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    ## model
    model = models.alexnet(pretrained=True)
    #model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True
    #model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])

    optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = 10**args.wd,
            )

    criterian = nn.CrossEntropyLoss().cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])


    tra = [transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]

    dir_names = open(args.img_dir_names, 'r').readlines()
    for index, dir_name in enumerate(dir_names):
        dir_name = dir_name.strip()
        num_img = len(glob.glob(os.path.join(args.root_dir, dir_name) + "/*." + args.extension))
        print(f"There are {num_img} images in {dir_name} directory")
        cluster_number = num_img // 2
        dataset = datasets.ImageFolder(args.root_dir , transform=transforms.Compose(tra))

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch,
            num_workers=1,
            pin_memory=True)


        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))
        print(features.shape)
        pca_model = PCA(n_components = 256)
        PCAed = pca_model.fit_transform(features)
        #kmeans = KMeans(n_clusters = cluster_number)
        #kmeans.fit(PCAed)
        #labels = kmeans.predict(PCAed)
        fcm_model = minmax.MinMaxKMeansMinCostFlow(cluster_number, size_min=2,   size_max=3)
        fcm_model.fit(PCAed)
        labels = fcm_model.predict(PCAed)
        with open(dir_name, 'w') as fp:
            for ind, label in enumerate(labels):
                new_line =  f"{args.root_dir}{dataset.imgs[ind][0].split('/')[-2]}/{dataset.imgs[ind][0].split('/')[-1]}:{dir_name}_{str(labels[ind])}\n"
                fp.write(new_line) 

if __name__ == '__main__':
    args = parse_args()
    main(args)
