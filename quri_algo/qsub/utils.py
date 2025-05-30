from math import pi

from quri_parts.qsub.lib.std import SWAP, Controlled, H, Phase
from quri_parts.qsub.op import Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.sub import SubBuilder


class _QFTdag(ParamUnitarySubDef[int]):
    name = "QFTdag"

    def qubit_count_fn(self, bits: int) -> int:
        return bits

    def sub(self, builder: SubBuilder, bits: int) -> None:
        qubits = builder.qubits

        for k in range(bits // 2):
            builder.add_op(SWAP, (qubits[k], qubits[bits - k - 1]))

        for k in range(bits):
            for i in range(k):
                builder.add_op(
                    Controlled(Phase(-pi / (2 ** (k - i)))), (qubits[i], qubits[k])
                )
            builder.add_op(H, (qubits[k],))


QFTdag, QFTdagSub = param_opsub(_QFTdag)


class _LineH(ParamUnitarySubDef[int]):
    name = "LineH"

    def qubit_count_fn(self, bits: int) -> int:
        return bits

    def sub(self, builder: SubBuilder, bits: int) -> None:
        qubits = builder.qubits

        for k in range(bits):
            builder.add_op(H, (qubits[k],))


LineH, LineHSub = param_opsub(_LineH)


class _QPE(ParamUnitarySubDef[int, Op]):
    name = "QPE"

    def qubit_count_fn(self, bits: int, op: Op) -> int:
        return bits + op.qubit_count

    def sub(self, builder: SubBuilder, bits: int, op: Op) -> None:
        qubits = builder.qubits
        pqs, sqs = qubits[:bits], qubits[bits:]

        builder.add_op(LineH(bits), pqs)
        for k in range(bits):
            for _ in range(2**k):
                builder.add_op(Controlled(op), (pqs[k], *sqs))
        builder.add_op(QFTdag(bits), pqs)


QPE, QPESub = param_opsub(_QPE)


class _QPEListUk(ParamUnitarySubDef[int, tuple[Op]]):
    name = "QPEListUk"

    def qubit_count_fn(self, bits: int, ops: tuple[Op]) -> int:
        return bits + ops[0].qubit_count

    def sub(self, builder: SubBuilder, bits: int, ops: tuple[Op]) -> None:
        qubits = builder.qubits
        pqs, sqs = qubits[:bits], qubits[bits:]

        builder.add_op(LineH(bits), pqs)
        for k in range(bits):
            builder.add_op(Controlled(ops[k]), (pqs[k], *sqs))
        builder.add_op(QFTdag(bits), pqs)


QPEListUk, QPEListUkSub = param_opsub(_QPEListUk)
