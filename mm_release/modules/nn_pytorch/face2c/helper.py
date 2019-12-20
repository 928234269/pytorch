import json
import time
import datetime


class TaskProcessor(object):
    def __init__(self, key, task, paid=False, upscale=4):
        super(TaskProcessor, self).__init__()
        self.key = key
        self.data = task
        self.code = 0
        self.err = ''
        self.paid = paid
        self.upscale = upscale

        self.faces = []
        self.quads = []

        self.face_diff = False

        self.is_gray = False
        self.orig_yuv = None

        self.processed_face_count = 0
        self.icc = None

    def input_url(self):
        return self.data['input_url']

    def input_fn(self):
        fn = f"{self.data['taskid']}_input"
        return fn

    def output_fn(self):
        fn = f"{self.data['taskid']}.jpg" if self.paid else \
            f"{self.data['taskid']}_diff.jpg"
        return fn

    def output_diff_fn(self, face_index):
        fn = f"{self.data['taskid']}_face_{face_index}.jpg"
        return fn

    def get_server_data(self):
        return json.dumps(self.data)

    def set_phase(self, p):
        self.data['phase'] = p

    def set_diff_url(self, url):
        self.data['diff_url'] = url

    def set_output_url(self, url):
        self.data['output_url'] = url

    def set_diff_data(self, data):
        self.data['diff_url'] = data

    def set_url(self, url):
        if self.paid:
            self.set_output_url(url)
            self.set_phase(7)
        else:
            self.set_diff_url(url)
            self.set_phase(3)

    def set_count(self, max_faces, all_faces):
        self.data['max_face_num'] = max_faces
        self.data['all_face_num'] = all_faces

    def set_author(self, server_id):
        self.data['author_id'] = server_id

    def is_sleep(self):
        return "sleep" in self.data

    def get_sleep_duration(self):
        return self.data['sleep']


class ServerInfo(object):
    def __init__(self, server_id, db):
        super(ServerInfo, self).__init__()
        self.db = db
        self.server_id = server_id
        self.server_key = f'server_{server_id}'
        self.server_data = dict(
            server_id=server_id,
            busy=0,
            # last_ping=int(time.time())
            last_ping=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def update_remote(self):
        # self.server_data['last_ping'] = int(time.time())
        self.server_data['last_ping'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.set(self.server_key, json.dumps(self.server_data))

    def set_busy(self, b, sync=False):
        self.server_data['busy'] = 1 if b else 0
        if sync:
            self.update_remote()

    def update(self):
        # now = int(time.time())
        now = datetime.datetime.now() #.timestamp()
        try:
            last_ping = datetime.datetime.strptime(self.server_data['last_ping'], "%Y-%m-%d %H:%M:%S")
        except TypeError as e:
            print("strptime TypeError", e)
            last_ping = now - datetime.timedelta(seconds=20)
        # if now - self.server_data['last_ping'] > 15:
        if (now - last_ping).total_seconds() > 15:
            self.update_remote()
            print(f"update server status, id {self.server_id} time {now}")




def diagnose_input(rgb2yuv, task, input):
    # grayscale check
    yuv = rgb2yuv.do(input)
    std = yuv[:, 1:, :, :].std()
    is_gray = std.item() < 0.012580

    print(f"uv std {std.item()} gray ? {is_gray}")

    if task is not None:
        task.is_gray = is_gray
        task.orig_yuv = yuv

    return is_gray


#fp16 helper
import torch
import collections
import torch.nn as nn
class tofp16(nn.Module):
    """
    Model wrapper that implements::
        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()

class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, *input):
        output, classes = input[0]
        ret = [c.float() for c in classes]
        return output.float(), ret

def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()), tofp32())
    # return BN_convert_float(network.half())

if __name__ == "__main__":
    print("helper start")

    import PIL.Image as Image
    from os import listdir
    from os.path import join, basename
    from torchvision import transforms as T
    import torch
    import torch.nn as nn
    from DGPT.Utils.Preprocess import RGB2YUV

    dn = 'd:/pytorch/face2c/assets'
    face_dn = 'd:/pytorch/face2c/faces'

    ff = [join(dn, x) for x in listdir(dn) if x.endswith('_input')]

    tf = T.ToTensor()
    pf = T.ToPILImage()
    rgb2yuv = RGB2YUV('cuda')

    for f in ff:
        bn = basename(f)
        try:
            img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"{bn} is not an image")
            continue

        input = tf(img).unsqueeze(0).to('cuda')

        is_gray = diagnose_input(rgb2yuv, None, input)

        if is_gray:
            print(f"{bn} is grayscale ")


class InferenceError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

import sys
class CUDNNRequest(object):
    REQ_TYPE_FACE = 1
    REQ_TYPE_BACKGROUD = 2

    def __init__(self, type, tensor):
        self.type = type
        self.tensor = tensor


import zmq
import uuid
class CUDNNRequestHandler(object):
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

        self.socket.connect("tcp://localhost:7801")

        client_id = uuid.uuid1().bytes
        self.socket.setsockopt(zmq.IDENTITY, client_id)

        self.poll = zmq.Poller()
        self.poll.register(self.socket, zmq.POLLIN)

        self.seq = 1

    def send_background_request(self, tensor):
        req = CUDNNRequest(CUDNNRequest.REQ_TYPE_BACKGROUD, tensor)
        self.send(req)

        return self.recv()


    def send_face_request(self, tensor):
        req = CUDNNRequest(CUDNNRequest.REQ_TYPE_FACE, tensor)
        self.send(req)

        return self.recv()

    def send(self, req):
        raw_data = req.tensor.half().cpu().numpy().tobytes()
        req_data_size = len(raw_data)

        aa = bytearray()
        aa.extend(self.seq.to_bytes(4, sys.byteorder))
        aa.extend(req.type.to_bytes(4, sys.byteorder))
        for d in req.tensor.shape:
            aa.extend(d.to_bytes(4, sys.byteorder))
        aa.extend(req_data_size.to_bytes(4, sys.byteorder))
        print("actual header len", len(aa))
        aa.extend(b'\0' * (128 - len(aa)))
        aa.extend(raw_data)

        self.socket.send(bytes(aa))

        self.seq += 1

    def recv(self):
        socks = dict(self.poll.poll(30000))
        if socks.get(self.socket) == zmq.POLLIN:
            message = self.socket.recv()
            print("Received reply [ %s ] header seq %d" % (len(message),
                                                              int.from_bytes(message[:4],
                                                                             sys.byteorder)))
            return message
        else:
            # print("W: No response from server, retrying…")
            # Socket is confused. Close and remove it.
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()

            self.poll.unregister(self.socket)

            print("E: Server seems to be offline, abandoning")

            raise InferenceError(TaskProcessor.ERROR_CUDNN_BACKEND_NO_RESPONSE, "cudnn backend timeout")

            return None
