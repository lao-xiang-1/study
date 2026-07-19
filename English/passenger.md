#en/Passenger
	This module provides a small set of utility functions for working with tree-like
data structures, such as nested tuples, lists, and dicts. We call these
structures pytrees. They are trees in that they are defined recursively (any
non-pytree is a pytree, i.e. a leaf, and any pytree of pytrees is a pytree) and
can be operated on recursively (object identity equivalence is not preserved by
mapping operations, and the structures cannot contain reference cycles).
	The set of Python types that are considered pytree nodes (e.g. that can be
mapped over, rather than treated as leaves) is extensible. There is a single
module-level registry of types, and class hierarchy is ignored. By registering a
new pytree node type, that type in effect becomes transparent to the utility
functions in this file.
	The primary purpose of this module is to enable the interoperability between
user defined data structures and JAX transformations (e.g. `jit`). This is not
meant to be a general purpose tree-like data structure handling library.
?
1. 本模块提供了一小组用于处理树状数据结构（tree-like data structures）的实用函数，这些数据结构包括嵌套元组、列表和字典。我们将这些结构称为 pytree。它们是树，因为它们是递归定义的（任何非 pytree 都是 pytree，即叶子节点；任何 pytree 的 pytree 也是 pytree），并且可以递归操作（映射操作不保留对象标识等价性，且这些结构不能包含引用循环）。
2. 被视为 pytree 节点（即可以被映射遍历，而非当作叶子节点处理）的 Python 类型集合是可扩展的。存在一个模块级别的类型注册表，且类继承层次结构被忽略。通过注册一个新的 pytree 节点类型，该类型实际上对本文件中的实用函数变为透明的。
3. 本模块的主要目的是实现用户自定义数据结构与 JAX 变换（如 `jit`）之间的互操作性。它并非一个通用的树状数据结构处理库。

