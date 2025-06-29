import os
import csv
import cv2
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from copy import deepcopy
from datetime import datetime
from dahuffman import HuffmanCodec
from memory_profiler import profile

import torch
import torch.utils.data
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import center_crop, resize

from hiner_utils import *
from data import HSIDataSet
from models import HINER, HINERDecoder, TransformInput


def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='hsi id',)
    parser.add_argument('--shuffle', action='store_true', help='randomly shuffle the mapping between frame idx and frame')
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='hsi crop size',)
    parser.add_argument('--ori_shape', type=str, default='640_1280', help='hsi ori size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='hsi resize size',)

    # architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--arch', type=str, default='hiner', help='model architecture')
    parser.add_argument('--data_type', type=str, default='hsi', help='datasets architecture')
    parser.add_argument('--embed', type=str, default='', help='empty string for HINER, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')

    # Decoding parameters: FC + Conv
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
        help='conv type for encoder/decoder', choices=['none', 'pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Epoch number')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='Fusion6', help='loss type, default=L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_image', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file',  default='', type=str, help='specify the embedding file')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")


    args = parser.parse_args()
    torch.set_printoptions(precision=4) 
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    if 'hiner' in args.arch:
        args.enc_strds = args.dec_strds
    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    extra_str = 'Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}'.format(args.modelsize, args.conv_type[0], args.enc_strd_str, 
        args.conv_type[1], args.dec_strd_str, '' if args.norm == 'none' else f'_{args.norm}', 
        '_dist' if args.distributed else '', '_shuffle_data' if args.shuffle_data else '',)
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    embed_str = f'{args.embed}_Dim{args.enc_dim}'
    exp_id = f'{args.vid}/{args.data_split}_{embed_str}_FC{args.fc_hw}_KS{args.ks}_RED{args.reduce}_low{args.lower_width}_blk{args.num_blks}' + \
            f'_e{args.epochs}_b{args.batchSize}_{args.quant_str}_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

def data_to_gpu(x, device):
    return x.to(device)

def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    args.metric_names = [
        'pred_seen_psnr', 'pred_seen_ssim', 'pred_seen_sam',
        'pred_unseen_psnr', 'pred_unseen_ssim', 'pred_unseen_sam',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_seen_sam', 
        'quant_unseen_psnr', 'quant_unseen_ssim', 'quant_unseen_sam',
        'CAM', 'RMSE']

    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    # setup dataloader
    full_dataset = HSIDataSet(args)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=(sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=False, worker_init_fn=worker_init_fn)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    args.dump_vis = args.dump_image

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Compute the parameter number
    if 'pe' in args.embed or 'le' in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
        embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param 
    c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
    args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())

    # Building model
    #model = HINER(args)
    model = HINER(args).cuda()

    ##### get model params and flops #####
    if local_rank in [0, None]:
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
        total_param = decoder_param + embed_param / 1e6
        args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
        param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M'
        print(f'{args}\n {model}\n {param_str}', flush=True)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'{param_str}\n')
        writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    #if args.distributed and args.ngpus_per_node > 1:
       # model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    #elif args.ngpus_per_node > 1:
      #  model = torch.nn.DataParallel(model)
    #elif torch.cuda.is_available():
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    args.transform_func = TransformInput(args)

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] 
        args.start_epoch = max(args.start_epoch, 0)

    # if args.eval_only:
      #  print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
       # results_list, hw = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
        #print_str = f'PSNR for output {hw} for quant {args.quant_str}: '
        #for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
         #   best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
          #  cur_v = RoundTensor(best_metric_value, 2 if 'psnr' in metric_name else 4)
           # print_str += f'best_{metric_name}: {cur_v} | '
           # best_metric_list[i] = best_metric_value
       # if local_rank in [0, None]:
          #  print(print_str, flush=True)
           # with open('{}/eval.txt'.format(args.outf), 'a') as f:
           #     f.write(print_str + '\n\n')        
            #args.train_time, args.cur_epoch = 0, args.epochs
            #Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')

        #return

####################################################################
    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        results_list, hw = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
        print_str = f'Metrics for output {hw} for quant {args.quant_str}:\n'
    
    # Group metrics by type (pred/quant) and category (seen/unseen)
        metric_groups = {
            'Predicted (Seen)': [0, 1, 2],      # pred_seen_psnr, pred_seen_ssim, pred_seen_sam
            'Predicted (Unseen)': [3, 4, 5],     # pred_unseen_psnr, pred_unseen_ssim, pred_unseen_sam
            'Quantized (Seen)': [6, 7, 8],       # quant_seen_psnr, quant_seen_ssim, quant_seen_sam
            'Quantized (Unseen)': [9, 10, 11]    # quant_unseen_psnr, quant_unseen_ssim, quant_unseen_sam
        }
    
        for group_name, metric_indices in metric_groups.items():
            print_str += f'\n{group_name}:\n'
            for i in metric_indices:
                metric_name = args.metric_names[i]
                metric_value = results_list[i].max()
                best_value = best_metric_list[i]
                round_num = 4 if 'ssim' in metric_name else 2
                print_str += f'  {metric_name}: {RoundTensor(metric_value, round_num)} (Best: {RoundTensor(best_value, round_num)})\n'
    
        if local_rank in [0, None]:
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')
            args.train_time, args.cur_epoch = 0, args.epochs
            Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')
        return

######################################################################

    # Training
    start = datetime.now()

    psnr_list = []
    for epoch in range(args.start_epoch, args.epochs):
        model.train()       
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        # iterate over dataloader
        device = next(model.parameters()).device
        for i, sample in enumerate(train_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
            if i > 10 and args.debug:
                break

            # forward and backward
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            img_out, _, _ = model(img_data, norm_idx)
            
            final_loss = loss_fn(img_out*inpaint_mask, img_gt*inpaint_mask, args.loss)# + 0.01 * CAM.mean()    
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt)) 
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(pred_psnr, 2))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = img_out.shape[-2:]
            writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            writer.add_scalar('Train/loss', final_loss, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            results_list, hw = evaluate(model, full_dataloader, local_rank, args, 
                args.dump_vis if epoch == args.epochs - 1 else False, 
                True if epoch == args.epochs - 1 else False)            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                print_str = f'Eval at epoch {epoch+1} for {hw}: '
                for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                    best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                    if 'psnr' in metric_name:
                        writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
                        writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
                        if metric_name == 'pred_seen_psnr':
                            psnr_list.append(metric_value.max())
                        print_str += f'{metric_name}: {RoundTensor(metric_value, 2)} | '
                    best_metric_list[i] = best_metric_value
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),   
        }    
        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if (epoch + 1) % args.epochs == 0:
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                Dump2CSV(args, best_metric_list, results_list, psnr_list, f'epoch{epoch+1}.csv')
                torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')
                if best_metric_list[0]==results_list[0]:
                    torch.save(save_checkpoint, f'{args.outf}/model_best.pth')

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


# Writing final results in CSV file
def Dump2CSV(args, best_results_list, results_list, psnr_list, filename='results.csv'):
    result_dict = {'Vid':args.vid, 'CurEpoch':args.cur_epoch, 'Time':args.train_time, 
        'FPS':args.fps, 'Split':args.data_split, 'Embed':args.embed, 'Crop': args.crop_list,
        'Resize':args.resize_list, 'Lr_type':args.lr_type, 'LR (E-3)': args.lr*1e3, 'Batch':args.batchSize,
        'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}', 
        'ModelSize': args.modelsize, 'Epoch':args.epochs, 'Loss':args.loss, 'Act':args.act, 'Norm':args.norm,
        'FC':args.fc_hw, 'Reduce':args.reduce, 'ENC_type':args.conv_type[0], 'ENC_strds':args.enc_strd_str, 'KS':args.ks,
        'enc_dim':args.enc_dim, 'DEC':args.conv_type[1], 'DEC_strds':args.dec_strd_str, 'lower_width':args.lower_width,
         'Quant':args.quant_str, 'bits/param':args.bits_per_param, 'bits/param w/ overhead':args.full_bits_per_param, 
        'bits/pixel':args.total_bpp, f'PSNR_list_{args.eval_freq}':','.join([RoundTensor(v, 2) for v in psnr_list]),}
    result_dict.update({f'best_{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, best_results_list)})
    result_dict.update({f'{k}':RoundTensor(v, 2 if 'psnr' in k else 4) for k,v in zip(args.metric_names, results_list) if 'pred' in k})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict,index=[0]).to_csv(csv_path)


@torch.no_grad()
# @profile
def evaluate(model, full_dataloader, local_rank, args, 
    dump_vis=False, huffman_coding=False):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]
    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f'{args.outf}/visualize_model' + ('_quant' if model_ind else '_orig')
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)       
            ori_shape_h = int(args.ori_shape.split('_')[0])
            ori_shape_w = int(args.ori_shape.split('_')[1])
            ori_shape_c = int(args.ori_shape.split('_')[2])
            final_mat = np.zeros((ori_shape_h,ori_shape_w,ori_shape_c))
            if ori_shape_h == 146 and ori_shape_w == 146 :
                final_mat = np.zeros((145,145,ori_shape_c)) 

        for i, sample in enumerate(full_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
            if i > 10 and args.debug:
                break
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            img_out, embed_list, dec_time = cur_model(img_data, norm_idx,  input_embed = 1 if model_ind else None)
            if model_ind == 0:
                img_embed_list.append(embed_list[0])
            
            # collect decoding fps
            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    img_out, embed_list, dec_time = cur_model(img_data, norm_idx, embed_list[0])
                    time_list.append(dec_time)

            # compute psnr and ms-ssim
            #pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
            pred_psnr = psnr_fn_batch([img_out], img_gt)
            pred_ssim = ssim_fn_batch([img_out], img_gt)  # Using SSIM instead of MS-SSIM
            pred_sam = sam_fn_batch([img_out], img_gt)
            for metric_idx, cur_v in enumerate([pred_psnr, pred_ssim, pred_sam]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 3 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start+metric_idx+6*model_ind].append(cur_v[:,batch_i])
            #for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
              #  for batch_i, cur_img_idx in enumerate(img_idx):
               #     metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                 #   metric_list[metric_idx_start+metric_idx+4*model_ind].append(cur_v[:,batch_i])
                        
            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    img_ind = cur_img_idx.cpu().numpy()
                    full_ind = i * args.batchSize + batch_ind
                    dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                    save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')
                    data_png = concat_img.to('cpu').numpy()
                    cv2.imwrite(f'{visual_dir}/pred_{img_ind:04d}_{temp_psnr_list}.png',(data_png.squeeze() * 65535. / 255.))
                    # img_data_png = img_data[batch_ind].to('cpu').numpy()
                    # cv2.imwrite(f'{visual_dir}/ori_{img_ind:04d}_{temp_psnr_list}.png',(img_data_png.squeeze() * 65535. / 255.))
                
                    
                    # sio.savemat(f'{visual_dir}/pred_{img_ind:04d}_{temp_psnr_list}.mat',{'hsi':img_out[batch_ind].cpu().squeeze(0).numpy()})
                    # sio.savemat(f'{visual_dir}/ori_{img_ind:04d}_{temp_psnr_list}.mat',{'hsi':img_data[batch_ind].cpu().squeeze(0).numpy()})
                    # sio.savemat(f'{visual_dir}/pred_2_{img_ind:04d}_{temp_psnr_list}.mat',{'hsi':concat_img.cpu().squeeze(0).numpy()})

                    img = center_crop(img_out[batch_ind].cpu().squeeze(0), (ori_shape_h, ori_shape_w))
                    img = img.numpy()
                    if img.shape[0] == 146 and img.shape[1] == 146 :
                        img = img[0:145,0:145]
                    final_mat[:,:,img_ind] = img

            # print eval results and add to log txt
            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                # print(sum(time_list))
                print_str = '[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, '.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i+1, len(full_dataloader), round(fps, 1))
                metric_name = ('quant' if model_ind else 'pred') + '_seen_psnr'
                for v_name, v_list in zip(args.metric_names, metric_list):
                    if metric_name in v_name:
                        cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                        print_str += f'{v_name}: {RoundTensor(cur_value, 2)} | '
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')
        
        # embedding quantization
        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0) 
            quant_embed, _ = quant_tensor(vid_embed, args.quant_embed_bit)

        # Collect results from 
        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        h,w = img_data.shape[-2:]
        cur_model.train()
        if args.distributed and args.ngpus_per_node > 1:
            for cur_v in results_list:
                cur_v = all_reduce([cur_v.to(local_rank)])

        if dump_vis:
            final_mat_path = os.path.join(visual_dir, 'final_mat')
            if not os.path.exists(final_mat_path):
                os.makedirs(final_mat_path)
            sio.savemat(f'{final_mat_path}/{args.quant_model_bit}_final.mat',{'hsi':final_mat})
        
    # dump quantized checkpoint, and decoder
    if local_rank in [0, None] and quant_ckt != None:
        quant_vid = {'embed': quant_embed, 'model': quant_ckt}
        torch.save(quant_vid, f'{args.outf}/quant_vid.pth')
        torch.jit.save(torch.jit.trace(HINERDecoder(model), (vid_embed[:2])), f'{args.outf}/img_decoder.pth')
        # huffman coding
        if huffman_coding:
            quant_v_list = quant_embed['quant'].flatten().tolist()
            tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = total_bits / len(quant_v_list)
            
            # including the overhead for min and scale storage, 
            total_bits += tmin_scale_len * 16               #(16bits for float16)
            args.full_bits_per_param = total_bits / len(quant_v_list)

            # bits per pixel
            args.total_bpp = total_bits / args.final_size / args.full_data_length
            print(f'After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}')
    # import pdb; pdb.set_trace; from IPython import embed; embed()     

    return results_list, (h,w)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt


if __name__ == '__main__':
    main()
