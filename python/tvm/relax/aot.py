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
# pylint: disable=invalid-name, redefined-builtin, no-else-return
"""The Relax virtual machine"""
from typing import Callable, List, Optional, Union, Dict

import tvm
from tvm import relax
from tvm.relay.backend import Executor, Runtime
from tvm.target import Target, make_compilation_config
from tvm.ir.module import IRModule

from tvm import (
    relax,
    IRModule,
)


def _assign_targets_to_relaxfuncs_irmodule(mod, target):
    """helper to assign target for PrimFunc in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, (tvm.relax.Function, tvm.tir.PrimFunc)):
            ret[global_var] = basefunc.with_attr("target", target)
    return ret


def lower(mod: tvm.IRModule) -> tvm.IRModule:
    passes = [
        relax.transform.ToNonDataflow(),
        relax.transform.CallTIRRewrite(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.ConvertRelaxMainToDPS(),
        relax.transform.Normalize(),
    ]
    seq = tvm.transform.Sequential(passes)
    return seq(mod)


def build(
    ir_mod,
    target=None,
    target_host=None,
    mod_name="default",
):
    executor = Executor("aot")
    runtime = Runtime("cpp")

    if not isinstance(ir_mod, IRModule):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    ctxt = tvm.transform.PassContext()
    config = make_compilation_config(ctxt, target, target_host)

    ir_mod = lower(ir_mod)
    ir_mod = _assign_targets_to_relaxfuncs_irmodule(ir_mod, Target(target))
    ir_mod = ir_mod.with_attr("executor", executor)
    ir_mod = ir_mod.with_attr("runtime", runtime)

    relax_build = tvm.get_global_func("relax.aot.build")
    runtime_mod = relax_build(
        ir_mod,
        mod_name,
        config,
        executor,
        runtime,
    )

    fcreate = tvm.get_global_func("tvm.aot_executor_factory.create")
    exec_mod = fcreate(runtime_mod, mod_name)
    return exec_mod
