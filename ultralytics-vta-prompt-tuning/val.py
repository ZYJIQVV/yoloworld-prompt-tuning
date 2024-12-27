import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
from ultralytics import YOLOWorld
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='H_All_Add_F_ZC_Data_copypaste_split_Other_datasets.yaml', help='data.yaml path')
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--epochs', type=int, default=80, help='epochs')
    parser.add_argument('--batch', type=int, default=15, help='batch size')
    parser.add_argument('--names', type=str, default='my_names', help='names')
    # parser.add_argument('--weight', type=str, default='/data/D/zyj/code/ultralytics-prompt-tuning-yoloworld/runs/detect/train26/weights/best.pt', help='weight')
    parser.add_argument('--weight', type=str, default='yolov8x-worldv2.pt', help='weight')
    args = parser.parse_args()

    data = f'../{args.data}'
    epochs = args.epochs
    device = args.device
    batch = args.batch
    names = args.names
    with open(f'../{names}', 'r') as f:
        names = f.read().strip().split('\n')
    weight = args.weight
    device = [int(x) for x in device.split(',')]
    # model = YOLOWorld('yolov8x-worldv2-my.yaml')
    model = YOLOWorld(weight)
    model.set_classes(names)
    model.val(
        data=data, 
        device=device, 
        batch=batch * len(device),
        )




