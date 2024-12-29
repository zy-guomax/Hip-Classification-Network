import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.network import SegNet, Prediction, Projection
from core.config import config
from core.scheduler import PolyScheduler
from core.function import train, inference
from dataset.dataloader import get_train_loader, get_valid_loader
from utils.utils import determine_device, save_checkpoint, create_logger, setup_seed, InfoNCELoss


def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK  # True
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # False
    cudnn.enabled = config.CUDNN.ENABLED

    model = nn.ModuleDict({
        'segnet': SegNet(),
        'projections': nn.ModuleList([
            Projection(nf) for nf in [256, 128, 64, 32]
        ]),
        'predictions': nn.ModuleList([
            Prediction() for _ in range(4)
        ])
    })
    if config.TRAIN.PARALLEL:  # only cuda is supported
        devices = config.TRAIN.DEVICES
        model = nn.DataParallel(model, devices).cuda(devices[0])  # data is transferred to cuda
    else:  # support cuda, mps and ... cpu (really?)
        device = determine_device()
        model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.95,
                          nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)  # update learning speed, the closer it is to the end, the closer lr is to 0
    # deep supervision weights, normalize sum to 1
    criterion = InfoNCELoss()
    # training data generator
    train_loader = get_train_loader()
    # validation dataset
    valid_loader = get_valid_loader()

    best_model = False
    best_perf = 100
    logger = create_logger('log', 'train.log')
    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optimizer.param_groups[0]['lr']))

        train(model, train_loader, optimizer, criterion, logger, config, epoch)
        scheduler.step()
        # running validation at every epoch is time consuming
        if epoch % config.VALIDATION_INTERVAL == 0:
            perf = inference(model['segnet'], valid_loader, criterion, logger, config)

            if perf < best_perf:
                best_perf = perf
                best_model = True
            else:
                best_model = False

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'perf': perf,
                'optimizer': optimizer.state_dict(),
            }, best_model, config.OUTPUT_DIR, filename='checkpoint.pth')


if __name__ == '__main__':
    main()
