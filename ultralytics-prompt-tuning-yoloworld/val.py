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
    parser.add_argument('--names', type=str, default='names', help='names')
    parser.add_argument('--split', type=str, default='val', help='split')
    # parser.add_argument('--weight', type=str, default='/data/D/zyj/code/ultralytics-prompt-tuning-yoloworld/runs/detect/train26/weights/best.pt', help='weight')
    parser.add_argument('--weight', type=str, default='yolov8x-worldv2.pt', help='weight')
    args = parser.parse_args()
    coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    my_names = ['ship', 'plane']
    data = args.data
    epochs = args.epochs
    device = args.device
    batch = args.batch
    try:
        names = globals()[args.names]
    except:
        with open(args.names, 'r') as f:
            names = f.read().strip().split('\n')
    weight = args.weight
    split = args.split
    device = [int(x) for x in device.split(',')]
    model = YOLOWorld('yolov8x-worldv2-my.yaml')
    model.load(weight)
    model.set_classes(names)
    model.val(
        data=data, 
        device=device, 
        batch=batch * len(device),
        split=split,
        )




