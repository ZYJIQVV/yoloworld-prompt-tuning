import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from ultralytics import YOLOWorld
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco.yaml', help='data yaml path')
    parser.add_argument('--cfg', type=str, default='yolov8x-worldv2-my-rand-vision-prompt.yaml', help='model yaml path')
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--epochs', type=int, default=80, help='epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--weight', type=str, default='yolov8x-worldv2.pt', help='weight')
    parser.add_argument('--names', type=str, default='coco.names', help='names')
    parser.add_argument('--resume', type=bool, default=False, help='Resume from previous checkpoint')
    parser.add_argument('--save_period', type=int, default=5, help='Few shot learning')
    parser.add_argument('--few_shot', type=int, default=None, help='Few shot learning')
    parser.add_argument('--project', type=str, default='runs', help='project name')
    parser.add_argument('--name', type=str, default='train', help='save name')
    parser.add_argument('--cfg_or_weight', type=str, default='cw', help='Way to load the model. "c" for loading from cfg, "w" for loading from weight and "cw" for loading from cfg and then loading weight')
    
    args = parser.parse_args()
    data = f'{args.data}'
    cfg = args.cfg
    device = args.device
    epochs = args.epochs
    batch = args.batch
    weight = args.weight
    names = args.names
    with open(f'{names}', 'r') as f:
        names = f.read().strip().split('\n')
    resume = args.resume
    save_period = args.save_period
    few_shot = args.few_shot
    project = args.project
    name = args.name
    cfg_or_weight = args.cfg_or_weight
    device = [int(x) for x in device.split(',')]
    if cfg_or_weight == 'c':
        model = YOLOWorld(cfg)
        resume=False
    elif cfg_or_weight == 'w':
        model = YOLOWorld(weight)
    elif cfg_or_weight == 'cw':
        model = YOLOWorld(cfg)
        model.load(weight)
        resume=False
    model.set_classes(names)
    model.train(
        data=data, 
        device=device, 
        epochs=epochs, 
        batch=batch * len(device), 
        freeze=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
        resume=resume,
        few_shot=few_shot,
        project=project,
        name=name,
        save_period=save_period,
        names=names,
        )



