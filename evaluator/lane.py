import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json

num_of_lane_gt=[0 for i in range(0,10)]
num_of_lane_pred=[0 for i in range(0,10)]

class Eval_Cfg():
    def __init__(self):
        self.eval_list = []
        return
    def sort_list(self):
        self.eval_list.sort(key=lambda data : data.acc)
        return
class Eval_data():
    def __init__(self):
        self.acc = 0
        self.pred_lane = 0
        self.gt_lane = 0
        self.filePath = ""
        return
class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        # print("---")
        # print(pred[0][10])
        # if False:
        # if True:
        #     for lane in pred:
        #         for idx in range(len(lane)):
        #             if lane[idx] != -2:
        #                 lane[idx] -= 10
        # print(pred[0][10])
        if any(len(p) != len(y_samples) for p in pred):
            print("-----------")
            print(len(y_samples))
            for p in pred:
                print(len(p))
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        cfg = Eval_Cfg()
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        # print(json_pred)

        f = open("Score.txt",'w')
        lane_num=0
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            # print(gt_lanes)
            # print(len(gt_lanes))
            num_of_lane_gt[len(gt_lanes)] +=1
            pl = len(pred_lanes)
            if pl>9:
                pl=9
            num_of_lane_pred[pl] +=1
            # return
            y_samples = gt['h_samples']
            # print("Lane Num = {}".format(lane_num))
            lane_num+=1
            try:
                ed = Eval_data()
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
                f.write("LANE : {} / {}  ACC : {: >5.4f}, FP : {: >0.3f}, FN : {: >0.3f}     FILENAME {} \n".format(len(gt_lanes), len(pred_lanes), a,p,n, pred['raw_file']))

                ed.acc = a
                ed.gt_lane = len(gt_lanes)
                ed.pred_lane = len(pred_lanes)
                ed.filePath = pred['raw_file']
                cfg.eval_list.append(ed)


            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        print(json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ]))
        return cfg


if __name__ == '__main__':
    import sys
    print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    print("GT")
    print(num_of_lane_gt)
    print(sum(num_of_lane_gt))
    print("PRED")
    print(num_of_lane_pred)
    print(sum(num_of_lane_pred))
    # try:
        # if len(sys.argv) != 3:
        #     raise Exception('Invalid input arguments')
        # print( LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    # except Exception as e:
    #     print( e.message)
    #     sys.exit(e.message)
