from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim

class Module():
    def __init__(self, name:str) -> None:
        self.name = name
        self.accuracy = []
        self.losses = []
        self.batch_num = []
        self.name = name
        self.acc = 0
        self.loss = 0
        self.total = 0
        self.accs = []
        self.tlosses = []

    def add(self, accuracy, loss, batch_size):
        self.accuracy.append(accuracy)
        self.losses.append(loss * batch_size)
        self.batch_num.append(batch_size)

    def handle(self,):
        patch_size = sum(self.batch_num)
        self.acc = sum(self.accuracy) / patch_size * 100
        self.loss = sum(self.losses) / patch_size
        self.total += patch_size
        # clear
        self.accuracy = []
        self.losses = []
        self.batch_num = []
        # add to history
        self.accs.append(self.acc)
        self.tlosses.append(self.loss)
    
    def __call__(self):
        if len(self.batch_num) != 0:
            self.handle()
        return self.accs, self.tlosses

class Drawer():
    def __init__(self, ) -> None:
        self.modules = {'train':Module('train'), 'val':Module('val'), 'eval':Module('eval')}
        self.epoch = 0
        self.infos = {}
        

    def add(self, name, accuracy, loss, batch_size):
        self.modules[name].add(accuracy, loss, batch_size)
        

    def _update(self):
        self.epoch += 1
        for name, module in self.modules.items():
            self.infos[name] = module()

    def broadcast(self, time):
        self._update()
        print('[Epoch %d] Loss (train/val/eval): %.6f/%.6f/%.6f' % (self.epoch, self.infos['train'][1][-1], self.infos['val'][1][-1], self.infos['eval'][1][-1]),
        ' Acc (train/val/eval): %.2f%%/%.2f%%/%.2f%%' % (self.infos['train'][0][-1], self.infos['val'][0][-1], self.infos['eval'][0][-1]),
        ' Epoch Time: %.2fs' % (time))

    def __call__(self, key:str = 'acc'):
        if 'loss' in key.lower():
            return self.infos['train'][1][-1], self.infos['val'][1][-1], self.infos['eval'][1][-1]
        return self.infos['train'][0][-1], self.infos['val'][0][-1], self.infos['eval'][0][-1]

    def draw(self, output_file, traget_folder = 'assert'):
        output_file = os.path.join(traget_folder, f'{output_file}_accuracy.jpg')
        # accuracy figure
        for name, info in self.infos.items():
            plt.plot(info[0], label=name)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(output_file)

        plt.figure()
        # loss figure
        for name, module in self.infos.items():
            plt.plot(module[1], label=name)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(output_file.replace('accuracy', 'loss'))
        plt.show()
        

if __name__ == '__main__':
    drawer = Drawer()
    for _ in range(10):
        for i in range(10):
            drawer.add('train', i*_/100, i, 100)
            drawer.add('val', i*_/100, i, 100)
            drawer.add('eval', i*_/100, i, 100)
        drawer.broadcast(1)
    print(drawer.infos)
    drawer.draw('test')