import torch
from utils.utils import AverageMeter, determine_device
from scipy.stats import norm


def D(p, z):
    return - torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()


def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    model.train()
    segnet = model['segnet']
    losses = AverageMeter()  # average loss of previous losses
    # if scaler is not supported, it switches to default mode, the training can also continue
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.TRAIN.NUM_BATCHES
    for idx in range(num_iter):
        norm_img, pos_img, neg_img = next(iter(train_generator))
        if config.TRAIN.PARALLEL:
            devices = config.TRAIN.DEVICES
            norm_img = norm_img.cuda(devices[0])
            pos_img = [img.cuda(devices[0]) for img in pos_img]
            neg_img = [img.cuda(devices[0]) for img in neg_img]
        else:
            devices = config.TRAIN.DEVICES
            norm_img = norm_img.cuda(devices[0])
            pos_img = [img.cuda(devices[0]) for img in pos_img]
            neg_img = [img.cuda(devices[0]) for img in neg_img]
        # run training
        with torch.cuda.amp.autocast():
            pos_out = []
            for img in pos_img:
                out = segnet(img, norm_img)
                pos_out.append(out)
            neg_out = []
            for img in neg_img:
                out = segnet(img, norm_img)
                neg_out.append(out)
            l_ce = criterion(pos_out, neg_out)
            loss = l_ce
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        torch.save(segnet.state_dict(), '/root/segnet.pt')


        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx, num_iter,
                    loss = losses,
                )
            logger.info(msg)


def inference(model, valid_dataset, criterion, logger, config):
    model.eval()    # model = model['segnet']
    norm_img, pos_img, neg_img = next(iter(valid_dataset))
    if config.TRAIN.PARALLEL:
        devices = config.TRAIN.DEVICES
        norm_img = norm_img.cuda(devices[0])
        pos_img = pos_img.cuda(devices[0])
        neg_img = [img.cuda(devices[0]) for img in neg_img]
    else:
        device = determine_device()
        norm_img = norm_img.cuda(device[0])
        pos_img = pos_img.cuda(device[0])
        neg_img = [img.cuda(device[0]) for img in neg_img]
    # run training
    with torch.no_grad():
        pos_out, embeds = model(pos_img, norm_img)
        neg_out, embeds = model(neg_img, norm_img)
        l_ce = criterion(pos_out, neg_out)
    logger.info('------------  CrossEntropy Loss ------------')
    logger.info(l_ce)
    return l_ce
