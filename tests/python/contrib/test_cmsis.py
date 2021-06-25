# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import tvm
from tvm import te
import numpy as np
import tvm.topi.testing
from tvm.contrib import cmsis
import tvm.testing

def verify_matmul_add(m, l, n, lib, transa=False, transb=False, dtype="float32"):
    bias = te.var("bias", dtype=dtype)
    ashape = (l, n) if transa else (n, l)
    bshape = (m, l) if transb else (l, m)
    A = te.placeholder(ashape, name="A", dtype=dtype)
    B = te.placeholder(bshape, name="B", dtype=dtype)
    C = lib.matmul(A, B, transa, transb)
    D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
    s = te.create_schedule(D.op)

    def get_numpy(a, b, bb, transa, transb):
        if transa:
            a = a.transpose()
        if transb:
            b = b.transpose()
        return np.dot(a, b) + bb

    def compile(f, name="test_matmul_add", ext=".so"):
        path = name + ext
        f.export_library(path)
        mod = tvm.runtime.load_module(path)
        f = mod[name]
        return f

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func(lib.__name__ + ".matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        name = "test_matmul_add"
        f = tvm.build(s, [A, B, D, bias], target, name=name)
        if target == "c":
            f = compile(f, name)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), dev)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), dev)
        bb = 10.0
        f(a, b, d, bb)
        tvm.testing.assert_allclose(
            d.numpy(), get_numpy(a.numpy(), b.numpy(), bb, transa, transb), rtol=1e-5
        )

    verify("c")

def test_matmul_add():
    verify_matmul_add(235, 128, 1024, cmsis)
