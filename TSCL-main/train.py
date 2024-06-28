import argparse, os, datetime, time
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torchvision.transforms as transforms
import random
from torchvision import datasets

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100

from datasets.tinyimages_300k import TinyImages
from models.resnet import ResNet18, ResNet34


from utils.utils import *
from utils.ltr_metrics import *
from utils.loss_fn import *

# to prevent PIL error from reading large images:
# See https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162#issuecomment-491115265
# or https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

class OODDataset(torch.utils.data.Dataset):
  def __init__(self, images, labels):
        self.labels = labels
        self.images = images

  def __len__(self):
        return len(self.images)

  def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y
def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='without  selected_ood training a (k+1) classifier with compact_loss  for OOD detection in long-tailed recognition')
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', '--cpus', type=int, default=4, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='/root/long_tailed_ood/datasets', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    # parser.add_argument('--quantile', default=0.125, type=float, help='quantile')
    # training params:
    parser.add_argument('--seed', type=int, default=110, help='seed=110 for cifar100-LT')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay_epochs', '--de', default=[60,80], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--Lambda', default=1.0, type=float, help='OE loss term tradeoff hyper-parameter')
    parser.add_argument('--Lambda2', default=0.1, type=float, help='Contrastive loss term tradeoff hyper-parameter')
    parser.add_argument('--T', default=0.1, type=float, help='Temperature in NT-Xent loss (contrastive loss)')
    parser.add_argument('--k', default=0.6, type=float, help='bottom-k classes are taken as tail class')  # 源代码0.4
    parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
    # 
    parser.add_argument('--timestamp', action='store_true', help='If true, attack time stamp after exp str')
    parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
    parser.add_argument('--save_root_path', '--srp', default='./Results', help='data root path')
    # ddp 
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    args = parser.parse_args()

    assert args.k>0, "When args.k==0, it is just the OE baseline."

    return args


def create_save_path():
    # mkdirs:
    decay_str = args.decay # cos
    if args.decay == 'multisteps':
        decay_str += '-'.join(map(str, args.decay_epochs)) 
    opt_str = args.opt  # adam
    if args.opt == 'sgd':
        opt_str += '-m%s' % args.momentum
    opt_str = 'e%d-b%d-%s-lr%s-wd%s-%s' % (args.epochs, args.batch_size, opt_str, args.lr, args.wd, decay_str)
    reweighting_fn_str = 'sign' 
    loss_str = 'Lambda%s-Lambda2%s-T%s-%s' % (args.Lambda, args.Lambda2, args.T, reweighting_fn_str)
    # loss_str += '-k%s-trial%s'% (args.k, args.trial)
    loss_str += '-k%s-seed%s'% (args.k, args.seed)
    exp_str = '%s_%s' % (opt_str, loss_str)
    if args.timestamp:
        exp_str += '_%s' % datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_str = '%s-%s-OOD%d' % (args.dataset, args.imbalance_ratio, args.num_ood_samples) 
    save_dir = os.path.join(args.save_root_path, 'LT_OOD_results', dataset_str, args.model, exp_str)
    # save_dir = os.path.join(args.save_root_path, 'LT_OOD_results', 'add_tail_class_centers_cifar100_ablation_k', dataset_str, args.model, exp_str)
    create_dir(save_dir)
    print('Saving to %s' % save_dir)

    return save_dir

def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(gpu_id, ngpus_per_node, args):  # train(0, 0, args)

    save_dir = args.save_dir

    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id  #  node_id=0

    print(f"Running on rank {rank}.")

    # Initializes ddp:
    if args.ddp: # 'If true, use distributed data parallel'
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda' # device =''cuda
    torch.backends.cudnn.benchmark = True

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        assert False, 'Not supported dataset: {}'.format(args.dataset)
   
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=TwoCropTransform(train_transform), imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        # test_set是cifar-10的正常test_set，即每个类1000张，共10000张
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=TwoCropTransform(train_transform), imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    else:
        assert False, 'Not supported dataset: {}'.format(args.dataset)

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=True)
    if args.dataset in ['cifar10', 'cifar100']:
        ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))  # num_ood_samples=30000, Number of OOD samples to use
        # ood_set = TinyImages(args.data_root_path, transform=train_transform)
    else:
        assert False, 'Not supported dataset: {}'.format(args.dataset)
   
    if args.ddp:
        ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set)
    else:
        ood_sampler = None
    ood_loader = DataLoader(ood_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=ood_sampler)
    print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))
    ood_dataset_size = len(train_loader.dataset) * 2   # ood_dataset_size = 12406*2=24812
    print('OOD Dataset Size: ', ood_dataset_size) #  OOD Dataset Size:  24812
    # Training on cifar10 with 12406 images and 10000 validation images | 30000 OOD training images.
    
    # get prior distributions:
    img_num_per_cls = np.array(train_set.img_num_per_cls) # eg,  img_num_per_cls= [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    img_num_per_cls = torch.from_numpy(img_num_per_cls).to(device)

    _sigmoid_x = torch.linspace(-1, 1, num_classes).to(device) # # _sigmoid_x= tensor([-1.0000, -0.7778, -0.5556, -0.3333, -0.1111,  0.1111,  0.3333,  0.5556,0.7778,  1.0000])
    _d = -2 * args.k + 1 - 0.001 # - 0.001 to make _d<-1 when k=1
    cl_loss_weights = torch.sign((_sigmoid_x-_d)) # k=0.4, 则cl_loss_weights= tensor([-1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.])
                                                                             # k=0.5, 则cl_loss_weights= tensor([-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.])
    plt.plot(cl_loss_weights.detach().cpu().numpy())
    plt.grid(True, which='both')
    plt.savefig(os.path.join(save_dir, 'cl_loss_weights.png'))
    plt.close()

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes+1, return_features=True).to(device)
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes+1, return_features=True).to(device)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)
    
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False)
    else:
        # model = torch.nn.DataParallel(model)
        pass

    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=True)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.decay == 'multisteps':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # train:
    if args.resume:  #'If true, resume from early stopped ckpt
        ckpt = torch.load(os.path.join(save_dir, 'latest.pth'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])  
        start_epoch = ckpt['epoch']+1 
        best_overall_acc = ckpt['best_overall_acc']
        training_losses = ckpt['training_losses']
        test_clean_losses = ckpt['test_clean_losses']
        f1s = ckpt['f1s']
        overall_accs = ckpt['overall_accs']
        many_accs = ckpt['many_accs']
        median_accs = ckpt['median_accs']
        low_accs = ckpt['low_accs']
    else:
        training_losses, test_clean_losses = [], []
        f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
        best_overall_acc = 0
        start_epoch = 0

    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+') # a+: 附加读写方式打开
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')
    ##############  starting training epochs  ########################
    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        start_time = time.time()

        model.train()
        training_loss_meter = AverageMeter()
        current_lr = scheduler.get_last_lr()
        
        ############ 加入NTOM #####################
        # selected_ood_loader = select_ood(ood_loader, model, args.batch_size , num_classes, ood_dataset_size, args.quantile)  # batch_size=256
        for batch_idx, ((in_data, labels), (ood_data, _)) in enumerate(zip(train_loader, ood_loader)):
        
            in_data = torch.cat([in_data[0], in_data[1]], dim=0) # shape=(2*N,C,H,W). Two views of each image.
            in_data, labels = in_data.to(device), labels.to(device)
            
            # print('in_data.shape=',in_data[1].shape)   # in_data[0].shape= torch.Size([256, 3, 32, 32])
            # print('batch_idx=',batch_idx)                    # 一直到batch_idx= 47   因为12406/256=48.46
            # print('labels.shape=',labels.shape)            # labels.shape= torch.Size([256])
            # print('ood_data.shape=',ood_data.shape) #ood_data.shape= torch.Size([256, 3, 32, 32])
            ###############################
            N_out = ood_data.shape[0]
            ood_data = ood_data.to(device)
            out_target = (torch.ones(N_out) * num_classes).long()  # 10000*10, tensor([10., 10., 10., 10., 10., ....])
            out_target = out_target.to(device)
            # print('out_target.shape',out_target.shape)  # 256
            ################################
            N_in = labels.shape[0]  # N_in=256
            
            all_data = torch.cat([in_data, ood_data], dim=0) # all_data.shape=torch.Size([2*Nin+Nout,C,W,H]=torch.Size([768, 3, 32, 32]
            
            # forward:
            all_f, all_logits, centers = model.forward_projection(all_data)   # all_f= feat_mlp
            # all_logits, p4 = model(all_data)  # p4是penultimate layer 的feature, p4.shape= torch.Size([768, 512]), all_logits.shape= torch.Size([768, 10])
            in_logits = all_logits[0:2*N_in]
            in_loss = F.cross_entropy(in_logits, torch.cat([labels, labels], dim=0))

            ood_logits = all_logits[2*N_in:]
            # ood_loss = oe_loss_fn(ood_logits)
            ood_loss = F.cross_entropy(ood_logits, out_target)

            # contrastive loss between tail-class and OOD samples:

            ground_truth = list(range(0,num_classes))  ##生成列表
            k_tail_labels = ground_truth[round((1-args.k)*num_classes):]  ## 提取出尾部类标签
            k_tail_labels = torch.tensor(k_tail_labels)
            k_tail_labels =k_tail_labels.to(device)
            # print('tail_labels=',k_tail_labels)  # tail_labels= tensor([5, 6, 7, 8, 9])
            # print('labels=',labels)
            tail_idx = labels>= round((1-args.k)*num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            # print('tail_idx=',tail_idx)   tail_idx元素全为 true 或者false
            # if args.ddp:  # args.k=0.4 即bottom-k classes are taken as tail class
            #     all_f = model.module.forward_projection(p4)
            # else:
            #     all_f = model.forward_projection(p4) # all_f.shape= torch.Size([768, 128])
            #     # print('all_f.shape=', all_f.shape)
            
            f_id_view1, f_id_view2 = all_f[0:N_in], all_f[N_in:2*N_in]  # f_id_view1.shape= torch.Size([256, 128])
            f_id_tail_view1 = f_id_view1[tail_idx] # i.e., 选出label=5,6,7,8,9 的iamge in cifar10
            # f_id_tail_view1.shape= torch.Size([num, 128])
            f_id_tail_view2 = f_id_view2[tail_idx] 
            labels_tail = labels[tail_idx]
            # print('labels_tail.shape=',labels_tail.shape)  # labels_tail.shape= torch.Size([24])
            # 选出尾部类质心
            centers_tail = centers[k_tail_labels]  # centers_tail.shape= torch.Size([5, 128])
            # print('centers_tail.shape=',centers_tail.shape)  # all_labels_tail.shape= torch.Size([29])

            # all_labels_tail = torch.cat([labels_tail, tail_labels], dim=0)
            # print('all_labels_tail.shape=',all_labels_tail.shape)  # all_labels_tail.shape= torch.Size([29])
            
            f_ood = all_f[2*N_in:]
            if torch.sum(tail_idx) > 0:
                cl_loss = my_cl_loss_fn(
                    torch.stack((f_id_tail_view1, f_id_tail_view2), dim=1), centers_tail, f_ood, labels_tail, k_tail_labels,temperature=args.T,
                    reweighting=True, w_list=cl_loss_weights
                )  #  f_id =torch.stack((f_id_tail_view1, f_id_tail_view2), dim=1)后，f_id.shape=torch.Size([N_id, 2, 128]),即将同一个样本两次数据增强后的两个样本打包在一起
            else:
                cl_loss = 0*ood_loss

            loss = in_loss + args.Lambda * ood_loss + args.Lambda2 * cl_loss

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            if rank == 0 and batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss:%.4f (in_loss:%.4f, ood_loss:%.4f, cl_loss:%.4f) | lr:%s' % (
                    epoch, batch_idx, loss.item(), in_loss.item(), ood_loss.item(), cl_loss.item(), current_lr) 
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        # lr update:
        scheduler.step()

        if rank == 0:
            # eval on clean set:
            model.eval()
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            preds_list, labels_list = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits, _ = model(data)
                    ############################################
                    pred = logits[:,:num_classes].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    ######################
                    loss = F.cross_entropy(logits, labels)
                    # test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                    test_loss_meter.append(loss.item())
                    preds_list.append(pred)
                    labels_list.append(labels)

            preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
            labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()

            overall_acc= (preds == labels).sum().item() / len(labels)
            f1 = f1_score(labels, preds, average='macro')

            many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)

            test_clean_losses.append(test_loss_meter.avg)
            f1s.append(f1)
            overall_accs.append(overall_acc)
            many_accs.append(many_acc)
            median_accs.append(median_acc)
            low_accs.append(low_acc)

            val_str = 'epoch %d (test): ACC overall_acc:%.4f (many_acc:%.4f, median_acc:%.4f, low_ac:%.4f) | F1:%.4f | time:%s' % (epoch, overall_acc, many_acc, median_acc, low_acc, f1, time.time()-start_time) 
            print(val_str)
            fp_val.write(val_str + '\n')
            fp_val.flush()

            # save curves:
            training_losses.append(training_loss_meter.avg)
            plt.plot(training_losses, 'b', label='training_losses')
            plt.plot(test_clean_losses, 'g', label='test_clean_losses')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'losses.png'))
            plt.close()

            plt.plot(overall_accs, 'm', label='overall_accs')
            if args.imbalance_ratio < 1:
                plt.plot(many_accs, 'r', label='many_accs')
                plt.plot(median_accs, 'g', label='median_accs')
                plt.plot(low_accs, 'b', label='low_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_accs.png'))
            plt.close()

            plt.plot(f1s, 'm', label='f1s')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_f1s.png'))
            plt.close()

            # save best model:
            if overall_accs[-1] > best_overall_acc:
                best_overall_acc = overall_accs[-1]
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_clean_acc.pth'))


            # save pth:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 
                'best_overall_acc': best_overall_acc,
                'training_losses': training_losses, 
                'test_clean_losses': test_clean_losses, 
                'f1s': f1s, 
                'overall_accs': overall_accs, 
                'many_accs': many_accs, 
                'median_accs': median_accs, 
                'low_accs': low_accs, 
                }, 
                os.path.join(save_dir, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_dir = create_save_path()
    args.save_dir = save_dir

    # 设置随机数种子
    setup_seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    
    # set CUDA:
    if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
        torch.cuda.device_count()
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.ddp: # If true, use distributed data parallel
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)