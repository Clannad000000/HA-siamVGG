from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC

from got10k.utils.metrics import rect_iou, center_error
import io

def _calc_metrics(boxes, anno):
    # can be modified by children classes
    ious = rect_iou(boxes, anno)
    center_errors = center_error(boxes, anno)
    return ious, center_errors

def _calc_curves(ious, center_errors):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, nbins_iou)[np.newaxis, :]
    thr_ce = np.arange(0, nbins_ce)[np.newaxis, :]

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)

    return succ_curve, prec_curve

seqs = ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2','BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4',
        'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds','David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
        'Football', 'Freeman4', 'Girl', 'Human3', 'Human4','Human6', 'Human9', 'Ironman', 'Jump', 'Jumping',
        'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam','Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing',
        'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis','Walking', 'Walking2', 'Woman',
        'Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board','Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon','Crossing',
        'Dancer', 'Dancer2', 'David2', 'David3','Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', 'Fish','FleetFace', 'Football1',
        'Freeman1', 'Freeman3','Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8','Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang',
        'MountainBike', 'Rubik', 'Singer1', 'Skater','Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans','Twinnings', 'Vase']

if __name__ == '__main__':
    nbins_iou = 21
    nbins_ce = 51
    video_path = 'E:\\xxx\\OTB2015\\Bolt'
    img_files = sorted(glob.glob(os.path.join(video_path, 'img/*.jpg')))
    anno_files = glob.glob(os.path.join(video_path, 'groundtruth_rect*.txt'))
    with open(anno_files[0], 'r') as f:
        anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

    net_path = './pretrained/model.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    boxes, _ ,fps= tracker.track(img_files, anno[0, :], visualize=True,debug = False, gt=anno)
    ious, center_errors = _calc_metrics(boxes, anno)
    succ_curve, prec_curve = _calc_curves(ious, center_errors)
    print('OP is {:.3f},DP is {:.3f},AUC is {:.3f},fps is {:.3f}'.format(len(ious[ious > 0.5]) / len(ious), prec_curve[20],np.mean(succ_curve),fps))

