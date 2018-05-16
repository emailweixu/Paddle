#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
import paddle
import unittest
import numpy


class TestNestedRNN(unittest.TestCase):
    def setUp(self):
        self.dict_dim = 10
        self.word_dim = 8
        self.hidden_dim = 8
        self.label_dim = 3
        self.data = [
            [[[1, 3, 2], [4, 5, 2]], 0],
            [[[0, 2], [2, 5], [0, 1, 2]], 1],
        ]

    def rnn_data(self):
        for d in self.data:
            seq = []
            for subseq in d[0]:
                seq += subseq
            yield seq, d[1]

    def nested_rnn_data(self):
        for d in self.data:
            yield d

    def rnn(self):
        data = fluid.layers.data(
            name='word', shape=[1], dtype='int64', lod_level=1)
        emb = fluid.layers.embedding(
            input=data, size=[self.dict_dim, self.word_dim])

        rnn = fluid.layers.DynamicRNN()

        with rnn.block():
            y = rnn.step_input(emb)
            mem = rnn.memory(shape=[self.hidden_dim])
            out = fluid.layers.fc(input=[y, mem],
                                  size=self.hidden_dim,
                                  act='tanh')
            rnn.update_memory(mem, out)
            rnn.output(out)

        rep = fluid.layers.sequence_last_step(input=rnn())
        prob = fluid.layers.fc(input=rep, size=self.label_dim, act='softmax')

        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        loss = fluid.layers.cross_entropy(prob, label)
        loss = fluid.layers.mean(loss)
        sgd = fluid.optimizer.Adam(1e-3)
        sgd.minimize(loss=loss)
        return [data, label], [loss]

    def nested_rnn(self):
        data = fluid.layers.data(
            name='word', shape=[1], dtype='int64', lod_level=1)
        emb = embedding_layer(input=data, size=word_dim)
        # not finished yet

    def test_rnn(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            inputs, outputs = self.rnn()

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=inputs, place=cpu)
        dataset = paddle.batch(self.rnn_data, batch_size=2)
        for data in dataset():
            print data
            loss_np = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=outputs)[0]


if __name__ == '__main__':
    unittest.main()
