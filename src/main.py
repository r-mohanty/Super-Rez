import torch

import utility
import data
import model
import copy
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        pass
        # from videotester import VideoTester
        # model = model.Model(args, checkpoint)
        # t = VideoTester(args, model, checkpoint)
        # t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            
            model_ema = copy.deepcopy(_model).eval()
            #model_ema.load(????)
            
            _loss = loss.Loss(args, checkpoint, _model) if not args.test_only else None
            t = Trainer(args, loader, _model, model_ema, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
