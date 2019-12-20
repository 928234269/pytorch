import os, sys

sys.path.append(os.getcwd())
sys.path.append("..")


class OnePageEngine(object):
    class Event(object):
        pass

    def __init__(self, opt):
        self.callbacks = {}
        self.bulletin_board = {}
        self.opt = opt
        self.count = 0

    def register(self, name, callback, replace=True):
        if name not in self.callbacks or replace is True:
            self.callbacks[name] = callback
        else:
            raise Exception(f"will overwrite callback {name}")

    def autoreg(self, name):
        def decorator(f):
            self.register(name, f)
        return decorator

    def fire(self, name, **attrs):
        if name not in self.callbacks.keys():
            return None

        e = OnePageEngine.Event()
        e.source = self
        for k, v in attrs.items():
            setattr(e, k, v)

        return self.callbacks[name](e)


    def dump_callbacks(self):
        for k in self.callbacks.keys():
            print("CB: ", k, self.callbacks[k])

    def record(self, key, value):
        self.bulletin_board[key] = value

    def play(self, key):
        if key in self.bulletin_board:
            return self.bulletin_board[key]
        return None

    def run(self):
        self.prepare()
        self.train()
        self.finish()

    def valid(self):
        self.prepare()
        self.epoch = 0
        self.fire('on_valid_all')
        self.finish()

    def preprocess(self):
        self.prepare()
        self.epoch = 0
        self.fire('on_test_preprocess')
        self.finish()

    def prepare(self):
        self.fire('on_prepare')

    def train(self):
        for epoch in range(self.opt.train.epochs):
            self.epoch = epoch

            self.fire('on_epoch_start')
            self.train_epoch()
            self.fire('on_epoch_end')


    def finish(self):
        self.fire('on_finish')


    def train_epoch(self):
        self.fire('on_train_epoch')


    def dataset_crop_callback(self, img):
        crop_size = self.fire('on_crop_callback', img=img)
        if crop_size is None:
            crop_size = self.opt.common.crop_size
        return crop_size


    def dataset_load_callback(self, img, transform=None):
        data = self.fire('on_load_callback', img=img, tf=transform, augment=True)

        return data
