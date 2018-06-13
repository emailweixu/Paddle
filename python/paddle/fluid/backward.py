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

from paddle.fluid import framework as framework
from . import core
import collections
import copy
import unique_name

__all__ = [
    'append_backward',
    'calc_gradient',
]


def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    for i in range(begin_idx, end_idx):
        op_desc = op_descs[i]
        if isinstance(op_desc, tuple):
            op_desc = op_desc[0]
        op_desc.rename_input(old_name, new_name)
        op_desc.rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    """
    Create a C++ OpDesc object with specified inputs, outputs and attributes.
    """
    op_desc = core.OpDesc()
    op_desc.set_type(op_type)
    for para, args in inputs.iteritems():
        op_desc.set_input(para, args)
    for para, args in outputs.iteritems():
        op_desc.set_output(para, args)
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

    if op_role_attr_name not in attrs:
        attrs[
            op_role_attr_name] = core.op_proto_and_checker_maker.OpRole.Backward
    for name, val in attrs.iteritems():
        if isinstance(val, framework.Block):
            op_desc.set_block_attr(name, val.desc)
        else:
            op_desc.set_attr(name, val)
    return op_desc


def _infer_var_data_type_(grad_var_name, block):
    """
    Infer the data type of given grad variable
    """
    grad_var = block.desc.find_var_recursive(grad_var_name.encode("ascii"))
    fwd_name = _strip_grad_suffix_(grad_var_name.encode("ascii"))
    if block.desc.has_var_recursive(fwd_name):
        fwd_var = block.desc.find_var_recursive(fwd_name.encode("ascii"))
        grad_var.set_dtype(fwd_var.dtype())
    else:
        grad_var.set_dtype(core.VarDesc.VarType.FP32)


def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given varibale name
    e.g. x@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
    """
    pos = name.find(core.grad_var_suffix())
    return name[:pos] if pos != -1 else name


def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return name + core.grad_var_suffix()


def _addup_repetitive_outputs_(op_descs):
    """
    In backward part, an variable may be the output of more than one ops.
    In this case, the variable should be the accumulation of all the outputs.
    `sum_op`s are added to implement the accumulate.
    """
    pending_sum_ops = []
    var_rename_count = collections.defaultdict(int)
    renamed_vars = collections.defaultdict(list)
    for idx, op_desc in enumerate(op_descs):
        for var_name in op_desc.input_arg_names():
            if len(renamed_vars[var_name]) > 1:
                pending_sum_ops.append(
                    (_create_op_desc_("sum", {"X": renamed_vars[var_name]},
                                      {"Out": [var_name]}, {}), idx))
                renamed_vars[var_name] = [var_name]
        for var_name in op_desc.output_arg_names():
            if var_name == core.empty_var_name(
            ) or var_name in op_desc.input_arg_names():
                # empty variable or inplace op
                continue
            if len(renamed_vars[var_name]) == 0:
                # it's the first time we get the variable
                renamed_vars[var_name] = [var_name]
            else:
                if len(renamed_vars[var_name]) == 1:
                    new_name = var_name + "@RENAME@" + \
                               str(var_rename_count[var_name])
                    var_rename_count[var_name] += 1
                    # rename original var_name
                    renamed_vars[var_name][0] = new_name
                    _rename_arg_(op_descs, var_name, new_name, 0, idx)
                    _rename_arg_(pending_sum_ops, var_name, new_name)

                new_name = var_name + "@RENAME@" + \
                           str(var_rename_count[var_name])
                var_rename_count[var_name] += 1
                op_desc.rename_output(var_name, new_name)
                renamed_vars[var_name].append(new_name)
    for var_name, inputs in renamed_vars.iteritems():
        if len(inputs) > 1:
            pending_sum_ops.append((_create_op_desc_(
                "sum", {"X": inputs}, {"Out": [var_name]}, {}), len(op_descs)))
    # sum_op descs are sorted according to their insert position
    for p in reversed(pending_sum_ops):
        op_descs.insert(p[1], p[0])

    return op_descs


def _callback_lookup_(op):
    """
    Only used in _append_backward_ops_
    Build and returns a callback function for certain op. For example

    parallel_do:           AllReduce

    :param op:
    :return: callback function
    """
    if op.type == 'parallel_do' and op.attr('use_nccl'):
        all_vars = op.block.vars
        param_names = set(op.input('parameters'))
        param_names = filter(lambda name: all_vars[name].stop_gradient is False,
                             param_names)
        param_grad_names = [n + "@GRAD" for n in param_names]

        class ParallelDoCallBack(object):
            def __init__(self, param_grad_names, parallel_scopes_name):
                self.has_inserted_nccl_init = False
                self.param_grad_names = param_grad_names
                self.parallel_scopes_name = parallel_scopes_name

            def __call__(self, block, context):
                if not self.has_inserted_nccl_init:
                    op_desc = _create_op_desc_(
                        "ncclInit",
                        {"parallel_scopes": self.parallel_scopes_name},
                        {"Communicator": ['nccl_com__do_not_change_']}, {})
                    block.program.global_block().desc.append_op().copy_from(
                        op_desc)
                    self.has_inserted_nccl_init = True

                current_op_desc = context["__current_op_desc__"]
                for o_param in current_op_desc.output_names():
                    for o_argu in current_op_desc.output(o_param):
                        if o_argu in self.param_grad_names:
                            allreduce_out_name = o_argu + "__nccl_all_reduce__"
                            op_desc = _create_op_desc_(
                                "ncclReduce",
                                {
                                    "X": [o_argu],
                                    "Communicator":
                                    ['nccl_com__do_not_change_']
                                },
                                {"Out": [allreduce_out_name]},
                                {"reduction": "ncclSum",
                                 "root": 0}, )
                            block.desc.append_op().copy_from(op_desc)

                            op_desc = _create_op_desc_(
                                "assign", {"X": [allreduce_out_name]},
                                {"Out": [o_argu]}, {})
                            block.desc.append_op().copy_from(op_desc)

        return ParallelDoCallBack(param_grad_names,
                                  op.output("parallel_scopes"))
    else:
        return None


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, collections.Sequence) else [x]


def _handle_empty_output_gradient_(grad_op_descs, grad_block, block_outputs,
                                   target_grad_map):
    """
    Check if the output gradients are present.
    If not, we have two options:
    a. Create a variable and fill it with zeros
    b. Change the argument name to @EMPTY@
    Whehter to use a or b depends on whether the grad op supports b.

    Return:
       (list[str]): newly create gradient variables for storing zero gradient
    """

    grad_vars = [] if grad_block.idx == 0 else block_outputs
    grad_vars = map(_append_grad_suffix_, grad_vars)
    grad_vars = set([target_grad_map.get(grad, grad) for grad in grad_vars])
    op_descs = []

    for grad_op_desc in grad_op_descs:
        for grad_var_name in grad_op_desc.input_arg_names():
            if grad_var_name.find(core.grad_var_suffix()) == -1:
                continue
            grad_var_name = grad_var_name.encode("ascii")
            if grad_var_name in grad_vars:
                continue

            original_var_name = _strip_grad_suffix_(grad_var_name)

            if original_var_name in block_outputs:
                continue

            grad_vars.add(grad_var_name)

            print "add fill_zeros_like(X=%s, Out=%s) for %s" % (
                original_var_name, grad_var_name, grad_op_desc.type())

            # TODO: support option b in the above comment
            op_desc = _create_op_desc_("fill_zeros_like",
                                       {"X": [original_var_name]},
                                       {"Out": [grad_var_name]}, {})
            op_descs.append(op_desc)

        op_descs.append(grad_op_desc)

        for grad_var_name in grad_op_desc.output_arg_names():
            grad_vars.add(grad_var_name)

    return op_descs


def _backward_sub_block(op, sub_block_inputs, sub_block_outputs, sub_block,
                        callbacks):
    """
    Generate the gradient block for a sub block.
    Return grad_sub_block
    """

    grad_sub_block = sub_block.program.create_block()

    new_callbacks = callbacks
    cb = _callback_lookup_(op)
    if cb is not None:
        if callbacks is None:
            new_callbacks = [cb]
        else:
            new_callbacks = callbacks + [cb]

    grad_sub_block.set_forward_block_idx(sub_block.idx)
    sub_block_no_grad_set = set()
    for var in sub_block.vars.itervalues():
        assert isinstance(var, framework.Variable)
        if var.stop_gradient:
            sub_block_no_grad_set.add(var.name)

    _append_backward_ops_(
        sub_block,
        grad_sub_block,
        sub_block_outputs,
        sub_block_inputs,
        [],  # zero_grad_ops
        sub_block_no_grad_set,
        {},  #target_grad_map
        new_callbacks)

    sub_block.program.rollback()

    return grad_sub_block


from paddle.fluid.proto import framework_pb2


def var_desc_str(var_desc):
    return framework_pb2.VarDesc.FromString(str(var_desc.serialize_to_string()))


# FIXME: have general mechanism for deciding whether an operator can
# propagate gradients from outputs to its inputs
# These ops does not backprop gradients.
_no_gradient_flow_op_ = ['fill_constant_batch_size_like', 'fill_zeros_like']


def _append_backward_ops_(
        block,
        grad_block,
        outputs,  # variable names
        inputs,  # variable names
        zero_grad_ops,  # op_descs for creating zero gradients
        no_grad_set,  # variable names without @GRAD@ suffix
        target_grad_map,  # map from outputs to their gradients
        callbacks=None):
    """
    Creating gradient ops and variables in grad_block
    Return:
        (dict[str:str]): a map from original variable name to its gradient
        variable name.
    """
    print "_append_backward_ops_"
    print "inputs: ", inputs
    print "outputs: ", outputs

    # map from gradient variable name to orignal variable name
    # for variables in block
    grad_to_var = {}

    op_path = _find_op_path_(block, outputs, inputs, no_grad_set)

    grad_op_descs = zero_grad_ops
    program = block.program
    for op, new_op_inputs, new_op_outputs in reversed(op_path):
        print op.type
        print "new_op_inputs: ", new_op_inputs
        print "new_op_outputs: ", new_op_outputs

        grad_sub_block_list = []
        # If the op has its own sub-block, deal with the sub-block first
        if op.has_attr("sub_block"):
            sub_block = program.block(op.block_attr("sub_block"))
            grad_sub_block = _backward_sub_block(
                op, new_op_inputs, new_op_outputs, sub_block, callbacks)
            grad_sub_block_list.append(grad_sub_block.desc)

        # Getting op's corresponding grad_op
        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            op.desc, no_grad_set, grad_sub_block_list)

        grad_op_descs.extend(grad_op_desc)
        grad_to_var.update(op_grad_to_var)

    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs)

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(grad_block, grad_op_descs, grad_to_var, target_grad_map)

    grad_op_descs = _handle_empty_output_gradient_(grad_op_descs, grad_block,
                                                   outputs, target_grad_map)

    var_to_grad = dict([(v, k) for k, v in grad_to_var.iteritems()])

    # append op_desc in grad_op_descs to grad_block
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for grad_op_desc in grad_op_descs:
        op_desc = grad_block.desc.append_op()
        op_desc.copy_from(grad_op_desc)
        op_desc.set_attr(op_role_attr_name, backward)
        grad_to_var["__current_op_desc__"] = op_desc
        if callbacks is not None:
            assert (isinstance(callbacks, list))
            for cb in callbacks:
                cb(block=grad_block, context=grad_to_var)

    print "leaving _append_backward_ops_"
    return var_to_grad


def _append_backward_vars_(block, start_op_idx, outputs):
    """
    Create new variables required by backward pass.

    Args:
        block(Block): the block where new variables will be created
        start_op_idx(int): Only variables required by ops in
            block.ops[start_op_idx : ] will be created
    """

    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        if op_desc.has_attr("sub_block"):
            sub_block = block.program.block(op_desc.block_attr("sub_block"))
            _append_backward_vars_(sub_block, 0, op_desc.output_arg_names())
        new_vars = set()
        # create new gradient variables
        for grad_var_name in op_desc.output_arg_names():
            grad_var_name = grad_var_name.encode("ascii")
            if block.desc.has_var_recursive(
                    grad_var_name) or grad_var_name == core.empty_var_name():
                continue
            v = block.desc.var(grad_var_name)
            new_vars.add(grad_var_name)
        # infer_shape and infer_type
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)
        # ncclInit dones't need to set data_type
        if op_desc.type() == 'ncclInit':
            continue
        for arg in op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_(arg, block)


def _rename_grad_(block, grad_op_descs, grad_to_var, target_grad_map):
    """
    Rename the input and output arguments of the ops so that they are
    different from existing variables in the block.
    """
    var_map = copy.copy(target_grad_map)
    for op_desc in grad_op_descs:
        for name in op_desc.input_arg_names():
            if name in var_map:
                op_desc.rename_input(name, var_map[name])

        for name in op_desc.output_arg_names():
            if block.desc.find_var(name.encode("ascii")):
                new_name = unique_name.generate(name)
                op_desc.rename_output(name, new_name)
                var_map[name] = new_name

    for g, ng in var_map.iteritems():
        if g in grad_to_var:
            grad_to_var[ng] = grad_to_var[g]
            grad_to_var.pop(g)


def _augment_no_grad_set_(no_grad_set, block):
    """
    Add the variable in block with stop_gradient==True to no_grad_set
    """
    if no_grad_set is None:
        no_grad_set = set()
    no_grad_set = copy.copy(no_grad_set)
    for var in block.vars.itervalues():
        assert isinstance(var, framework.Variable)
        if var.stop_gradient:
            no_grad_set.add(var.name)

    return no_grad_set


def _set_op_role_var_attr_(program, params_and_grads):
    """
    For ops which outputs gradient, need to set the attribute
    "op_role_var" (kOpRoleVarAttrName()) to record the pair (param, grad)
    """
    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    for p, g in params_and_grads:
        if g is None:
            continue
        for op in reversed(program.global_block().ops):
            assert isinstance(op, framework.Operator)
            if g.name in op.output_arg_names:
                g.op = op
                break

        if g.op is None:
            raise ValueError("Unexpected branch")
        attr_val = [p.name, g.name]
        if g.op.has_attr(op_role_var_attr_name):
            attr_val.extend(g.op.attr(op_role_var_attr_name))
        g.op.set_attr(op_role_var_attr_name, attr_val)


def append_backward(loss, parameter_list=None, no_grad_set=None,
                    callbacks=None):
    """
    Append backward part to main_program

    Args:
        loss(Variable): The variable generated by cost function.
            It is must be in root block.
        parameter_list(list[string]): Parameters that need gradients.
            None means all parameters need gradients.
        no_grad_set(set): Variables that have no gradients in Block 0.
            All variables with `stop_gradient=True` from all blocks will be
            automatically added.

    Return:
        (list[(Variable,Variable)]): list of (parameter, gradient) pair.
    """
    assert isinstance(loss, framework.Variable)

    if loss.op is None:
        # the loss is from a cloned program. Find loss op manually.
        for op in reversed(loss.block.ops):
            assert isinstance(op, framework.Operator)
            if len(op.output_arg_names) == 1 and op.output_arg_names[
                    0] == loss.name:
                loss.op = op
                break
            if loss.op is None:
                raise ValueError("loss.op is None. Should not happend")

    loss.op.set_attr(core.op_proto_and_checker_maker.kOpRoleAttrName(),
                     int(core.op_proto_and_checker_maker.OpRole.Forward) |
                     int(core.op_proto_and_checker_maker.OpRole.Loss))

    if callbacks is not None:
        isinstance(callbacks, list)

    program = loss.block.program
    root_block = program.block(0)
    assert loss.block.idx == root_block.idx

    no_grad_set = _augment_no_grad_set_(no_grad_set, root_block)

    params = program.global_block().all_parameters()
    if parameter_list is not None:
        parameters = parameter_list
        for param in params:
            if param.name not in parameters:
                no_grad_set.add(param.name)
    else:
        parameters = [param.name for param in params]

    op_desc = _create_op_desc_(
        "fill_constant", {}, {"Out": [_append_grad_suffix_(loss.name)]}, {
            "shape": [1],
            "value": 1.0,
            "dtype": loss.dtype,
            "force_cpu": False,
            core.op_proto_and_checker_maker.kOpRoleAttrName():
            int(core.op_proto_and_checker_maker.OpRole.Backward) |
            int(core.op_proto_and_checker_maker.OpRole.Loss),
        })
    zero_grad_ops = [op_desc]

    start_op_idx = root_block.desc.op_size()
    var_to_grad = _append_backward_ops_(
        root_block,
        root_block,  # grad block is root block
        [loss.name],
        [],
        zero_grad_ops,
        no_grad_set,
        {},  # target_grad_map
        callbacks)
    _append_backward_vars_(root_block, start_op_idx, [loss.name])

    program.sync_with_cpp()

    # FIXME(zcd): prevent loss.grad optimized by mem_opt.
    loss.block.var(_append_grad_suffix_(loss.name)).persistable = True

    params_and_grads = []
    for param in parameters:
        if param not in var_to_grad:
            continue
        if not root_block.has_var(var_to_grad[param]):
            raise ValueError("BUG: root block does not have grad var %s" %
                             var_to_grad[param])
        # Get the param var from the global block
        param_var = root_block.var(param)
        grad_var = root_block.var(var_to_grad[param])
        params_and_grads.append((param_var, grad_var))

    _set_op_role_var_attr_(program, params_and_grads)

    return params_and_grads


def _find_op_path_(block, outputs, inputs, no_grad_set):
    """
    Find the ops on the path from inputs to outputs. For each such op,
    also find out all the inputs and output on the path.
    Returns:
        list[(op_desc, new_inputs, new_outputs)]
    """
    input_names = set(inputs)
    output_names = set(outputs)

    relevant_op_flags = [True] * len(block.ops)
    new_op_inputs = [[]] * len(block.ops)
    new_op_outputs = [[]] * len(block.ops)

    for i, op in enumerate(block.ops):
        if op.type in _no_gradient_flow_op_:
            relevant_op_flags[i] = False
            continue
        new_inputs = filter(lambda v: v not in no_grad_set,
                            op.desc.input_arg_names())
        if inputs:
            new_inputs = filter(lambda v: v in input_names, new_inputs)
        if new_inputs:
            for name in op.desc.output_arg_names():
                if name not in no_grad_set:
                    input_names.add(name)
            new_op_inputs[i] = new_inputs
        else:
            relevant_op_flags[i] = False

    for i, op in reversed(list(enumerate(block.ops))):
        if op.type in _no_gradient_flow_op_:
            continue
        new_outputs = filter(
            lambda v: v in output_names and v not in no_grad_set,
            op.desc.output_arg_names())
        if new_outputs:
            for name in op.desc.input_arg_names():
                if name not in no_grad_set:
                    output_names.add(name)
            new_op_outputs[i] = new_outputs
        else:
            relevant_op_flags[i] = False

    op_path = [(block.ops[i], new_op_inputs[i], new_op_outputs[i])
               for i in range(len(block.ops)) if relevant_op_flags[i]]
    return op_path


def calc_gradient(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    Backpropagate the graidents of targets to inputs.

    Args:
        targets(Variable|list[Variable]): The target variables
        inputs(Variable|list[Variable]): The input variables
        target_gradients(None | Variable | list[None|Variable]):
            The gradient for the targets.
            If is None, each target gradient is assumed to be a tensor of
            value one with the same shape as the corresponding target.
        no_grad_set(set[string]): The names of variables that have no gradients
            in Block 0. All variables with `stop_gradient=True` from all blocks
            will be automatically added.

    Return:
        (Variable | None | list[Variable|None]): list of gradients for inputs
        If an input does not affect targets, the corresponding gradient variable
        will be None
    """
    targets = _as_list(targets)
    inputs = _as_list(inputs)
    target_gradients = _as_list(target_gradients)

    block = targets[0].block
    prog = block.program
    block_idx = block.idx
    for target in targets:
        assert target.block.program == prog, \
            "All the targets should be from the same program"
        assert target.block.idx == block_idx, \
            "All the targets should be from the same block"

    for input in inputs:
        assert input.block.program == prog, \
            "input should be from the same program as targets"
        assert input.block.idx == block_idx or input.block.idx == 0, \
            "Input should be from the same block as targets"

    if not target_gradients:
        target_gradients = [None] * len(targets)

    assert len(targets) == len(target_gradients), \
        "Should have the same number of target_gradients as targets"

    no_grad_set = _augment_no_grad_set_(no_grad_set, block)

    target_grad_map = {}
    zero_grad_ops = []
    for i, grad in enumerate(target_gradients):
        target = targets[i]
        if grad is None:
            grad_name = _append_grad_suffix_(target.name)
            op_desc = _create_op_desc_("fill_constant_batch_size_like",
                                       {"Input": [target.name]},
                                       {"Out": [grad_name]}, {
                                           "shape": target.shape,
                                           "value": 1.0,
                                           "dtype": target.dtype,
                                           'input_dim_idx': 0,
                                           'output_dim_idx': 0
                                       })
            zero_grad_ops.append(op_desc)
        else:
            assert target.shape == grad.shape, \
                "The shapes of target and grad are different: %s %s" % (
                    target.name, grad.name)
            target_grad_map[_append_grad_suffix_(target.name)] = grad.name

    target_names = [target.name for target in targets]
    input_names = [input.name for input in inputs]
    start_op_idx = block.desc.op_size()
    var_to_grad = _append_backward_ops_(
        block,
        block,  # grad block is the same as block
        target_names,
        input_names,
        zero_grad_ops,
        no_grad_set,
        target_grad_map)
    _append_backward_vars_(block, start_op_idx, target_names)

    prog.sync_with_cpp()

    grad_vars = []
    for input_var in inputs:
        if input_var.name not in var_to_grad:
            grad_vars.append(None)
        else:
            assert block.has_var(var_to_grad[input_var.name]), \
                "BUG: cannot find grad var %s"  % var_to_grad[input_var.name]
            grad_vars.append(block.var(var_to_grad[input_var.name]))

    if len(grad_vars) == 1:
        return grad_vars[0]
    elif len(grad_vars) == 0:
        return None
    else:
        return grad_vars
