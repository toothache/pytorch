import torch
import torch.nn as nn

from sync_batchnorm import patch_replication_callback, SynchronizedBatchNorm2d
from models.resnet import resnet18


class MiniBatchDataParallel(nn.DataParallel):
    def __init__(self, module, cpu_keywords=[], **kwargs):
        super().__init__(module, **kwargs)
        self.cpu_keywords = cpu_keywords


    def forward(self, *inputs, **kwargs):
        inputs_list, kwargs_list = [], []
        for i, device_id in enumerate(self.device_ids):
            mini_inputs = [x[i] for x in inputs]
            mini_kwargs = dict([(k, v[i]) for k, v in kwargs.items()])
            a, b = self._minibatch_scatter(device_id, *mini_inputs, **mini_kwargs)
            inputs_list.append(a)
            kwargs_list.append(b)
        inputs = inputs_list
        kwargs = kwargs_list
        if len(self.device_ids) == 1:
            outputs = [self.module(*inputs[0], **kwargs[0])]
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

        # return [self.gather([x], self.output_device) for x in outputs]

    def _minibatch_scatter(self, device_id, *inputs, **kwargs):
        kwargs_cpu = {}
        for k in kwargs:
            if k in self.cpu_keywords:
                kwargs_cpu[k] = kwargs[k]
        for k in self.cpu_keywords:
            kwargs.pop(k, None)
        inputs, kwargs = self.scatter(inputs, kwargs, [device_id])
        kwargs_cpu = [kwargs_cpu] # a list of dict
        # Merge cpu kwargs with gpu kwargs
        for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
            d_gpu.update(d_cpu)
        return inputs[0], kwargs[0]


def main():
    """Main function"""
    resnet = resnet18(norm_layer=SynchronizedBatchNorm2d) # torchvision.models.resnet18()
    print(resnet)
    resnet.cuda()
    resnet = MiniBatchDataParallel(resnet)
    patch_replication_callback(resnet)

    tensor = torch.ones(())
    resnet.train()
    outputs = resnet([tensor.new_empty((4, 3, 1024, 320)), tensor.new_empty((4, 3, 960, 240))])
    print(type(outputs))
    print(len(outputs))


if __name__ == '__main__':
    main()
