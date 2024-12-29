import torch
from utils.utils import AverageMeter, determine_device


def D(p, z):
    return - torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()


def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    model.train()
    segnet = model['segnet']
    mlp = model['mlp']
    pretrained_path = '/root/autodl-fs/segnet.pt'
    seg_params = torch.load(pretrained_path)
    segnet.load_state_dict(seg_params, strict=False)
    for param in segnet.parameters():
        param.requires_grad = False
    losses = AverageMeter()  # average loss of previous losses
    # if scaler is not supported, it switches to default mode, the training can also continue
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.TRAIN.NUM_BATCHES
    for idx in range(num_iter):
        data, label = next(iter(train_generator))
        if config.TRAIN.PARALLEL:
            devices = config.TRAIN.DEVICES
            data = [train_data.cuda(devices[0]) for train_data in data]
            label = label.cuda(devices[0])
        else:
            devices = config.TRAIN.DEVICES
            data = [train_data.cuda(devices[0]) for train_data in data]
            label = label.cuda(devices[0])
        # run training
        with torch.cuda.amp.autocast():
            output = segnet(data[0], data[1])
            output = mlp(output)
            label = label.view(-1, 1)
            loss = criterion(output, label)
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()


        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx, num_iter,
                    loss = losses,
                )
            logger.info(msg)
    torch.save(mlp.state_dict(), '/root/mlp.pt')


def inference(model, valid_dataset, criterion, logger, config):
    model.eval()    # model = model['segnet']
    segnet = model['segnet']
    mlp = model['mlp']
    data, label = next(iter(valid_dataset))
    if config.TRAIN.PARALLEL:
        devices = config.TRAIN.DEVICES
        data = [train_data.cuda(devices[0]) for train_data in data]
        label = label.cuda(devices[0])
    else:
        devices = config.TRAIN.DEVICES
        data = [train_data.cuda(devices[0]) for train_data in data]
        label = label.cuda(devices[0])
    # run training
    with torch.no_grad():
        output = segnet(data[0], data[1])
        output = mlp(output)
        label = label.view(-1, 1)
        loss = criterion(output, label)
    logger.info('------------  CrossEntropy Loss ------------')
    logger.info(loss)
    return loss
