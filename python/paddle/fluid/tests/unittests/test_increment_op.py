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

import op_test
import numpy
import unittest


class TestIncrementOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "increment"
        x = numpy.random.random(size=1)
        self.inputs = {'X': x}
        self.outputs = {'Out': x + 1}

    def test_forward(self):
        self.check_output()

    def test_backward(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
