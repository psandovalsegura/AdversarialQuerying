import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from models.classification_heads import one_hot
from models.classification_heads import ClassificationHead
from models.protonet_embedding import ProtoNetEmbedding
from models.autoprotonet import AutoProtoNetEmbedding
from models.classification_heads import computeGramMatrix

def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def show_interpolation(embedding_net, source_data, prototype, nb_interpolate=10):
    """
    source_data is an image tensor
    prototype is an embedding 
    """
    prototype = prototype.view(1, -1)
    emb_source = embedding_net(source_data)
    emb_source = emb_source.view(1, -1)
    interpolants_imgs = []
    prototypes = []
    for i in range(nb_interpolate):
        d = emb_source - prototype
        new_prototype = prototype + (i * 1.0 / nb_interpolate) * d
        img_prototype = embedding_net.forward_decoder(new_prototype.view(embedding_net.embedding_shape))
        interpolants_imgs.append(img_prototype)
        prototypes.append(new_prototype)

    interpolants_imgs = torch.cat(interpolants_imgs).detach().cpu()
    return prototypes, interpolants_imgs

def support_query_data_labels(dataset, way, shot, query, epoch):
    (_, dataset_val, data_loader) = get_dataset(dataset)
    episodes_per_batch = 1
    val_episode = 2000
    dloader_val = data_loader(
            dataset=dataset_val,
            nKnovel=way,
            nKbase=0,
            nExemplars=shot,             # num training examples per novel category
            nTestNovel=query * way, # num test examples for all the novel categories
            nTestBase=0,                     # num test examples for all the base categories
            batch_size=episodes_per_batch,
            num_workers=0,
            epoch_size=1 * val_episode,  # num of batches per epoch
        )
    
    # For episodes_per_batch=1 after reshapes:
    # data_support torch.Size([25, 3, 32, 32])
    # labels_support torch.Size([1, 25])
    # data_query torch.Size([25, 3, 32, 32])
    # labels_query torch.Size([1, 25])
    batch = iter(dloader_val(epoch)).next()
    data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
    data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
    labels_support = labels_support.view(-1)
    data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
    labels_query = labels_query.view(-1)
    return data_support, labels_support, data_query, labels_query


# def get_support_query_data_labels_model(opt, epoch=0):
#     (_, dataset_val, data_loader) = get_dataset(opt)
#     episodes_per_batch = 1
#     val_episode = 2000
#     dloader_val = data_loader(
#             dataset=dataset_val,
#             nKnovel=opt.test_way,
#             nKbase=0,
#             nExemplars=opt.val_shot,             # num training examples per novel category
#             nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
#             nTestBase=0,                     # num test examples for all the base categories
#             batch_size=episodes_per_batch,
#             num_workers=0,
#             epoch_size=1 * val_episode,  # num of batches per epoch
#         )
    
#     # For episodes_per_batch=1 after reshapes:
#     # data_support torch.Size([25, 3, 32, 32])
#     # labels_support torch.Size([1, 25])
#     # data_query torch.Size([25, 3, 32, 32])
#     # labels_query torch.Size([1, 25])
#     batch = iter(dloader_val(epoch)).next()
#     data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
#     data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
#     data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))

#     embedding_net, cls_head = get_model(opt)
#     return data_support, labels_support, data_query, labels_query, embedding_net, cls_head

def get_model(network, dataset, ckpt_path=None, activation='LeakyReLU'):
    # Choose the embedding network
    if network == 'ProtoNet':
        network = ProtoNetEmbedding(activation = activation).cuda()
    elif network == 'AutoProtoNet':
        is_miniimagenet = dataset == 'miniImageNet'
        network = AutoProtoNetEmbedding(activation = activation, is_miniimagenet=is_miniimagenet).cuda()
    else:
        print ("Cannot recognize the network type {}")
        assert(False)
        
    # Choose the classification head
    cls_head = ClassificationHead(base_learner='ProtoNet').cuda()

    # Load from ckpt
    if ckpt_path:
        saved_model_ckpt = torch.load(os.path.join(ckpt_path, 'best_model.pth'))
        network.load_state_dict(saved_model_ckpt['embedding'])
        network.eval()
        if 'head' in saved_model_ckpt.keys():
            cls_head.load_state_dict(saved_model_ckpt['head'])
            cls_head.eval()

    return (network, cls_head)

def get_dataset(dataset):
    # Choose the embedding network
    if dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

def logits_using_prototypes(prototypes, query, n_way, n_shot, normalize=True):
    tasks_per_batch = query.size(0)
    d = query.size(2)
    
    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    
    if normalize:
        logits = logits / d

    return logits

def get_prototypes(support, support_labels, n_way, n_shot):
    """
    query: embedded query of shape (opt.episodes_per_batch, train_n_query, -1)
    support: embedded support of shape (opt.episodes_per_batch, train_n_support, -1)
    n_way: num train way
    n_shot: num train shot
    """
    tasks_per_batch = 1
    n_support = support.size(1)
    
    
    assert(support.dim() == 3)
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )
    return prototypes

def get_prototypes_from_data(embedding_net, data_support, labels_support, way, shot):
    with torch.no_grad():
        emb_support = embedding_net(data_support)
        train_n_support = way * shot
        episodes_per_batch = 1
        emb_support = emb_support.reshape(episodes_per_batch, train_n_support, -1)
        prototypes = get_prototypes(emb_support, labels_support, way, shot)
    return prototypes

def get_prototypes_imgs_from_data(embedding_net, data_support, labels_support, way, shot):
    prototypes = get_prototypes_from_data(embedding_net, data_support, labels_support, way, shot)
    prototypes_imgs = embedding_net.forward_decoder(prototypes.view(embedding_net.embedding_shape)).detach().cpu()
    return prototypes_imgs

def generate_prototype_img_using_adam(model, support, img_size, lr, 
                                      max_steps=200, rand_init=False, use_tv=False, start_from=None):
    """
    Optimizes an image which is close to the prototypes in L2
    Loss = |f(x) - p|_2
    
    prototype: the mean embedding of some support (1, 1600)
    """
    num_support = support.size(0)
    if rand_init:
        img = (torch.rand((num_support, 3, img_size, img_size), device='cuda')).requires_grad_(True)
    else:
        img = (torch.zeros((num_support, 3, img_size, img_size), device='cuda')).requires_grad_(True)
    
    if start_from is not None:
        img = start_from.detach().clone()
        img = img.cuda().requires_grad_(True)

    optimizer = torch.optim.Adam([img], lr=lr, betas=(0.9, 0.999), eps=1e-08)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                                           mode='min', 
    #                                                           factor=0.8, # factor by which lr is reduced
    #                                                           patience=100, 
    #                                                           threshold=0.001)
    
    for i in range(max_steps):
        img_embedding = model(img)
        assert img_embedding.shape == support.shape
        tv_loss = 1.0 if not use_tv else (0.05 * total_variation(img).mean())
        loss = torch.norm((img_embedding - support), p=2) + tv_loss
        
        optimizer.zero_grad()
        loss.backward()
        if i % 500 == 0:
            print(f'[Step {i}] Last LR: {optimizer.param_groups[0]["lr"]} L2 norm: {loss.item()}')
        optimizer.step()
        lr_scheduler.step(loss.item())
        
        with torch.no_grad():
            img = img.clamp_(0, 1) 
    
    return img

def generate_prototype_img_using_adam_xent(model, cls_head, emb_support, labels_support, img_size, lr, way, shot,
                                           max_steps=200, rand_init=False, use_tv=False, start_from=None):
    """
    Optimizes an image which is minimizes the loss to the support
    Loss = Xent(f(x), y)
    
    prototype: the mean embedding of some support (1, 1600)
    """
    num_support = emb_support.size(0)
    if rand_init:
        img = (torch.rand((num_support, 3, img_size, img_size), device='cuda')).requires_grad_(True)
    else:
        img = (torch.zeros((num_support, 3, img_size, img_size), device='cuda')).requires_grad_(True)
    
    if start_from is not None:
        img = start_from.detach().clone()
        img = img.cuda().requires_grad_(True)

    optimizer = torch.optim.Adam([img], lr=lr, betas=(0.9, 0.999), eps=1e-08)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                              mode='min', 
                                                              factor=0.8, # factor by which lr is reduced
                                                              patience=100, 
                                                              threshold=0.001)
    # create labels tensor for support
    
    labels_query = torch.ones((num_support), dtype=torch.long).cuda()
    for i in range(num_support):
        labels_query[i] = i
    
    for i in range(max_steps):
        emb_query = model(img)

        logit_query = cls_head(emb_query.reshape(1, way * num_support, -1), emb_support, labels_support, way, shot)
        smoothed_one_hot = one_hot(labels_query.reshape(-1), way)
        log_prb = F.log_softmax(logit_query.reshape(-1, way), dim=1)
        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
        
        tv_loss = 1.0 if not use_tv else (0.05 * total_variation(img).mean())
        loss = loss + tv_loss
        
        optimizer.zero_grad()
        loss.backward()
        if i % 500 == 0:
            print(f'[Step {i}] Last LR: {optimizer.param_groups[0]["lr"]} L2 norm: {loss.item()}')
        optimizer.step()
        lr_scheduler.step(loss.item())
        
        with torch.no_grad():
            img = img.clamp_(0, 1) 
    
    return img