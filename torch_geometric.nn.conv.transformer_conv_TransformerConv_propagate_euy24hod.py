import typing
from typing import Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.utils import is_sparse
from torch_geometric.typing import Size, SparseTensor

from torch_geometric.nn.conv.transformer_conv import *


from typing import List, NamedTuple, Optional, Union

import torch
from torch import Tensor

from torch_geometric.utils import is_torch_sparse_tensor
from torch_geometric.utils.sparse import ptr2index
from torch_geometric.typing import SparseTensor


class CollectArgs(NamedTuple):
    query_i: Tensor
    key_j: Tensor
    value_j: Tensor
    edge_attr: Optional[Tensor]
    index: Tensor
    ptr: Optional[Tensor]
    size_i: Optional[int]
    dim_size: Optional[int]


def collect(
    self,
    edge_index: Union[Tensor, SparseTensor],
    query: Tensor,
    key: Tensor,
    value: Tensor,
    edge_attr: OptTensor,
    size: List[Optional[int]],
) -> CollectArgs:

    i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

    # Collect special arguments:
    if isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            adj_t = edge_index
            if adj_t.layout == torch.sparse_coo:
                edge_index_i = adj_t.indices()[0]
                edge_index_j = adj_t.indices()[1]
                ptr = None
            elif adj_t.layout == torch.sparse_csr:
                ptr = adj_t.crow_indices()
                edge_index_j = adj_t.col_indices()
                edge_index_i = ptr2index(ptr, output_size=edge_index_j.numel())
            else:
                raise ValueError(f"Received invalid layout '{adj_t.layout}'")
            if edge_attr is None:
                _value = adj_t.values()
                edge_attr = None if _value.dim() == 1 else _value

        else:
            edge_index_i = edge_index[i]
            edge_index_j = edge_index[j]
            ptr = None

    elif isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        edge_index_i, edge_index_j, _value = adj_t.coo()
        ptr, _, _ = adj_t.csr()
        if edge_attr is None:
            edge_attr = None if _value is None or _value.dim() == 1 else _value

    else:
        raise NotImplementedError

    # Collect user-defined arguments:
    # (1) - Collect `query_i`:
    if isinstance(query, (tuple, list)):
        assert len(query) == 2
        _query_0, _query_1 = query[0], query[1]
        if isinstance(_query_0, Tensor):
            self._set_size(size, 0, _query_0)
        if isinstance(_query_1, Tensor):
            self._set_size(size, 1, _query_1)
            query_i = self._index_select(_query_1, edge_index_i)
        else:
            query_i = None
    elif isinstance(query, Tensor):
        self._set_size(size, i, query)
        query_i = self._index_select(query, edge_index_i)
    else:
        query_i = None
    # (2) - Collect `key_j`:
    if isinstance(key, (tuple, list)):
        assert len(key) == 2
        _key_0, _key_1 = key[0], key[1]
        if isinstance(_key_0, Tensor):
            self._set_size(size, 0, _key_0)
            key_j = self._index_select(_key_0, edge_index_j)
        else:
            key_j = None
        if isinstance(_key_1, Tensor):
            self._set_size(size, 1, _key_1)
    elif isinstance(key, Tensor):
        self._set_size(size, j, key)
        key_j = self._index_select(key, edge_index_j)
    else:
        key_j = None
    # (3) - Collect `value_j`:
    if isinstance(value, (tuple, list)):
        assert len(value) == 2
        _value_0, _value_1 = value[0], value[1]
        if isinstance(_value_0, Tensor):
            self._set_size(size, 0, _value_0)
            value_j = self._index_select(_value_0, edge_index_j)
        else:
            value_j = None
        if isinstance(_value_1, Tensor):
            self._set_size(size, 1, _value_1)
    elif isinstance(value, Tensor):
        self._set_size(size, j, value)
        value_j = self._index_select(value, edge_index_j)
    else:
        value_j = None

    # Collect default arguments:

    index = edge_index_i
    size_i = size[i] if size[i] is not None else size[j]
    size_j = size[j] if size[j] is not None else size[i]
    dim_size = size_i

    return CollectArgs(
        query_i,
        key_j,
        value_j,
        edge_attr,
        index,
        ptr,
        size_i,
        dim_size,
    )


def propagate(
    self,
    edge_index: Union[Tensor, SparseTensor],
    query: Tensor,
    key: Tensor,
    value: Tensor,
    edge_attr: OptTensor,
    size: Size = None,
) -> Tensor:

    # Begin Propagate Forward Pre Hook #########################################
    if not torch.jit.is_scripting() and not is_compiling():
        for hook in self._propagate_forward_pre_hooks.values():
            hook_kwargs = dict(
                query=query,
                key=key,
                value=value,
                edge_attr=edge_attr,
            )
            res = hook(self, (edge_index, size, hook_kwargs))
            if res is not None:
                edge_index, size, hook_kwargs = res
                query = hook_kwargs['query']
                key = hook_kwargs['key']
                value = hook_kwargs['value']
                edge_attr = hook_kwargs['edge_attr']
    # End Propagate Forward Pre Hook ###########################################

    mutable_size = self._check_input(edge_index, size)
    fuse = is_sparse(edge_index) and self.fuse

    if fuse:
        raise NotImplementedError("'message_and_aggregate' not implemented")

    else:

        kwargs = self.collect(
            edge_index,
            query,
            key,
            value,
            edge_attr,
            mutable_size,
        )

        # Begin Message Forward Pre Hook #######################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._message_forward_pre_hooks.values():
                hook_kwargs = dict(
                    query_i=kwargs.query_i,
                    key_j=kwargs.key_j,
                    value_j=kwargs.value_j,
                    edge_attr=kwargs.edge_attr,
                    index=kwargs.index,
                    ptr=kwargs.ptr,
                    size_i=kwargs.size_i,
                )
                res = hook(self, (hook_kwargs, ))
                hook_kwargs = res[0] if isinstance(res, tuple) else res
                if res is not None:
                    kwargs = CollectArgs(
                        query_i=hook_kwargs['query_i'],
                        key_j=hook_kwargs['key_j'],
                        value_j=hook_kwargs['value_j'],
                        edge_attr=hook_kwargs['edge_attr'],
                        index=hook_kwargs['index'],
                        ptr=hook_kwargs['ptr'],
                        size_i=hook_kwargs['size_i'],
                        dim_size=kwargs.dim_size,
                    )
        # End Message Forward Pre Hook #########################################

        out = self.message(
            query_i=kwargs.query_i,
            key_j=kwargs.key_j,
            value_j=kwargs.value_j,
            edge_attr=kwargs.edge_attr,
            index=kwargs.index,
            ptr=kwargs.ptr,
            size_i=kwargs.size_i,
        )

        # Begin Message Forward Hook ###########################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._message_forward_hooks.values():
                hook_kwargs = dict(
                    query_i=kwargs.query_i,
                    key_j=kwargs.key_j,
                    value_j=kwargs.value_j,
                    edge_attr=kwargs.edge_attr,
                    index=kwargs.index,
                    ptr=kwargs.ptr,
                    size_i=kwargs.size_i,
                )
                res = hook(self, (hook_kwargs, ), out)
                out = res if res is not None else out
        # End Message Forward Hook #############################################

        # Begin Aggregate Forward Pre Hook #####################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._aggregate_forward_pre_hooks.values():
                hook_kwargs = dict(
                    index=kwargs.index,
                    ptr=kwargs.ptr,
                    dim_size=kwargs.dim_size,
                )
                res = hook(self, (hook_kwargs, ))
                hook_kwargs = res[0] if isinstance(res, tuple) else res
                if res is not None:
                    kwargs = CollectArgs(
                        query_i=kwargs.query_i,
                        key_j=kwargs.key_j,
                        value_j=kwargs.value_j,
                        edge_attr=kwargs.edge_attr,
                        index=hook_kwargs['index'],
                        ptr=hook_kwargs['ptr'],
                        size_i=kwargs.size_i,
                        dim_size=hook_kwargs['dim_size'],
                    )
        # End Aggregate Forward Pre Hook #######################################

        out = self.aggregate(
            out,
            index=kwargs.index,
            ptr=kwargs.ptr,
            dim_size=kwargs.dim_size,
        )

        # Begin Aggregate Forward Hook #########################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._aggregate_forward_hooks.values():
                hook_kwargs = dict(
                    index=kwargs.index,
                    ptr=kwargs.ptr,
                    dim_size=kwargs.dim_size,
                )
                res = hook(self, (hook_kwargs, ), out)
                out = res if res is not None else out
        # End Aggregate Forward Hook ###########################################

        out = self.update(
            out,
        )

    # Begin Propagate Forward Hook ############################################
    if not torch.jit.is_scripting() and not is_compiling():
        for hook in self._propagate_forward_hooks.values():
            hook_kwargs = dict(
                query=query,
                key=key,
                value=value,
                edge_attr=edge_attr,
            )
            res = hook(self, (edge_index, mutable_size, hook_kwargs), out)
            out = res if res is not None else out
    # End Propagate Forward Hook ##############################################

    return out