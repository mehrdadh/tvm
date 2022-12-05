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
import numpy as np

import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.script import relax as R
from tvm.relax.aot import build
from tvm.relax.testing import relay_translator
from tvm.testing.aot import generate_ref_data
from tvm.relay import testing
from tvm.relay.frontend import from_onnx, from_tensorflow


def _export_mod(mod):
    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11", "-g3", "-O0"])
    return tvm.runtime.load_module(test_so_path)


def test_single_elementwise():
    dtype = "int32"
    target = "llvm"
    inputs = {"x": np.array([[-10, 5], [1, 2]], dtype=dtype)}

    def _relay():
        x = relay.var("x", shape=(2, 2), dtype=dtype)
        out = relay.abs(x)
        return relay.Function(relay.analysis.free_vars(out), out)

    def _reference(inputs):
        x = inputs["x"]
        return np.abs(x)  # abs

    relax_mod = relay_translator.from_relay(
        _relay(),
        target,
    )

    mod = build(relax_mod, target)
    loaded_mod = _export_mod(mod)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).numpy() == _reference(inputs)).all()


def test_scalar_constant():
    dtype = "int32"
    target = "llvm"
    inputs = {"x": np.array([[-10, 5], [1, 2]], dtype=dtype)}

    def _relay():
        x = relay.var("x", shape=(2, 2), dtype=dtype)
        out = relay.add(x, relay.const(-1, dtype=dtype))
        return relay.Function(relay.analysis.free_vars(out), out)

    def _reference(inputs):
        x = inputs["x"]
        return np.add(x, -1)  # add

    relax_mod = relay_translator.from_relay(
        _relay(),
        target,
    )

    mod = build(relax_mod, target)
    loaded_mod = _export_mod(mod)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).numpy() == _reference(inputs)).all()


def test_tensor_constant():
    dtype = "int32"
    target = "llvm"
    inputs = {"x": np.array([[-10, 1], [5, 1]], dtype=dtype)}

    def _relay():
        x = relay.var("x", shape=(2, 2), dtype=dtype)
        out = relay.add(x, relay.const(np.array([[1, 2], [3, 4]], dtype=dtype), dtype=dtype))
        return relay.Function(relay.analysis.free_vars(out), out)

    def _reference(inputs):
        x = inputs["x"]
        return np.add(x, np.array([[1, 2], [3, 4]]))  # add

    relax_mod = relay_translator.from_relay(
        _relay(),
        target,
    )

    mod = build(relax_mod, target)
    loaded_mod = _export_mod(mod)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).numpy() == _reference(inputs)).all()


def test_multi_input():
    dtype = "int32"
    target = "llvm"
    inputs = {
        "x": np.array([[-10, 1], [5, 1]], dtype=dtype),
        "y": np.array([[1, 2], [3, 4]], dtype=dtype),
    }

    def _relay():
        x = relay.var("x", shape=(2, 2), dtype=dtype)
        y = relay.var("y", shape=(2, 2), dtype=dtype)
        out = relay.add(x, y)
        return relay.Function(relay.analysis.free_vars(out), out)

    def _reference(inputs):
        x = inputs["x"]
        y = inputs["y"]
        return np.add(x, y)  # add

    relax_mod = relay_translator.from_relay(
        _relay(),
        target,
    )

    mod = build(relax_mod, target)
    loaded_mod = _export_mod(mod)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).numpy() == _reference(inputs)).all()


def test_multi_output():
    dtype = "int32"
    target = "llvm"
    inputs = {"x": np.array([[-10, 1], [5, 1]], dtype=dtype)}

    def _relay():
        x = relay.var("x", shape=(2, 2), dtype=dtype)
        abs = relay.abs(x)
        out = relay.subtract(abs, relay.const(1))
        out = relay.Tuple([abs, out])
        return relay.Function(relay.analysis.free_vars(out), out)

    def _reference(inputs):
        x = inputs["x"]
        abs = np.abs(x)  # abs
        out = abs - 1
        return [abs, out]

    relax_mod = relay_translator.from_relay(
        _relay(),
        target,
    )

    mod = build(relax_mod, target)
    loaded_mod = _export_mod(mod)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    for i, ref in enumerate(_reference(inputs)):
        assert (runner.get_output(i).numpy() == ref).all()


if __name__ == "__main__":
    tvm.testing.main()
