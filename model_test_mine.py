from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC
import traceback

class ExperimentOTB_model_test(ExperimentOTB):
    def __init__(self,root_dir,result_dir):
        super(ExperimentOTB_model_test, self).__init__(root_dir = root_dir,result_dir = result_dir)
        self.root_dir = root_dir
        self.result_dir = result_dir

    def run(self,tracker, visualize=False):
        tracker.name = self.result_dir.split('\\')[-1]
        print('Running tracker %s on %s...' % (tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(self.result_dir + '_siamvgg','%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue
            try:
                # tracking loop
                boxes, times, _ = tracker.track(img_files, anno[0, :], visualize=visualize)
                assert len(boxes) == len(anno)

                # record results
                self._record(record_file, boxes, times)
            except Exception as e:
                #print(e)
                with open('log.txt','w') as f:
                    traceback.print_exc(file=f)
                    #f.flush()
                    f.write(tracker.name)
                    f.close()
                break

if __name__ == '__main__':
    all_cp = []
    for file in os.listdir('E:\\xxx\\pretrained'):
        cp = os.path.join('E:\\xxx\\pretrained', file)
        all_cp.append(cp)

    all_cp.sort(key=lambda x: int(x.split('_e')[1][:-4]))

    for net_path in all_cp[::-1]:
        tracker = TrackerSiamFC(net_path=net_path)


        e = ExperimentOTB_model_test(root_dir = 'E:\\xxx\\OTB2015\\',
                                     result_dir=os.path.join('results', 'OTB2015', net_path.split('\\')[xxx].split('.')[0]))
        e.run(tracker)



