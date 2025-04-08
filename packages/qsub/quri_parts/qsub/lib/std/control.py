import math
from collections.abc import Collection

from quri_parts.qsub.op import (
    AbstractOp,
    Ident,
    Op,
    OpFactory,
    ParameterValidationError,
    ParamUnitaryDef,
    param_op,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.resolve import (
    SubRepository,
    SubResolver,
    SubResolverCondition,
    default_repository,
)
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS
from .cnot import CNOT
from .cz import CZ
from .rotation import RX, RY, RZ, Phase
from .single_clifford import H, S, Sdag, SqrtX, SqrtXdag, SqrtY, SqrtYdag, X, Y, Z
from .swap import SWAP
from .t import T, Tdag
from .toffoli import Toffoli

# Parametric Op definitions


class _Controlled(ParamUnitaryDef[Op]):
    ns = NS
    name = "Controlled"

    def qubit_count_fn(self, target_op: Op) -> int:
        return target_op.qubit_count + 1

    def reg_count_fn(self, target_op: Op) -> int:
        return target_op.reg_count

    def validate_params(self, target_op: Op) -> None:
        if not target_op.unitary:
            raise ParameterValidationError(f"target_op {target_op} is not unitary")


Controlled: OpFactory[Op] = param_op(_Controlled)


class _MultiControlled(ParamUnitaryDef[Op, int, int]):
    ns = NS
    name = "MultiControlled"

    def qubit_count_fn(
        self, target_op: Op, control_bits: int, control_value: int
    ) -> int:
        return target_op.qubit_count + control_bits

    def reg_count_fn(self, target_op: Op, control_bits: int, control_value: int) -> int:
        return target_op.reg_count

    def validate_params(
        self, target_op: Op, control_bits: int, control_value: int
    ) -> None:
        if not target_op.unitary:
            raise ParameterValidationError(f"target_op {target_op} is not unitary")
        if not control_bits >= 1:
            raise ParameterValidationError(
                f"control_bits should be a positive integer but {control_bits}"
            )
        if not (control_value >= 0 and control_value < 2**control_bits):
            raise ParameterValidationError(
                f"control_value should be a {control_bits}-bits integer but"
                f"{control_value}"
            )


MultiControlled: OpFactory[Op, int, int] = param_op(_MultiControlled)


# Sub resolver definitions


def controlled_sub_resolver(op: Op, repository: SubRepository) -> Sub | None:
    target_op = op.id.params[0]
    assert isinstance(target_op, Op)

    target_sub_resolver = repository.find_resolver(target_op)
    if not target_sub_resolver:
        return None
    target_sub = target_sub_resolver(target_op, repository)
    if not target_sub:
        return None

    builder = SubBuilder(op.qubit_count, op.reg_count)
    c, *target_q = builder.qubits

    target_aq = tuple(builder.add_aux_qubit() for _ in target_sub.aux_qubits)
    qubit_map = dict(zip(target_sub.qubits, target_q)) | dict(
        zip(target_sub.aux_qubits, target_aq)
    )

    target_ar = tuple(builder.add_aux_register() for _ in target_sub.aux_registers)
    reg_map = dict(zip(target_sub.registers, builder.registers)) | dict(
        zip(target_sub.aux_registers, target_ar)
    )

    for o, qs, rs in target_sub.operations:
        if o.unitary:
            builder.add_op(
                Controlled(o),
                (c, *(qubit_map[q] for q in qs)),
                tuple(reg_map[r] for r in rs),
            )
        else:
            builder.add_op(
                o,
                tuple(qubit_map[q] for q in qs),
                tuple(reg_map[r] for r in rs),
            )

    phase = target_sub.phase % (2 * math.pi)
    if phase == math.pi:
        builder.add_op(Z, (c,))
    elif phase == math.pi / 2:
        builder.add_op(S, (c,))
    elif phase == 3 * math.pi / 2:
        builder.add_op(Sdag, (c,))
    elif phase != 0:
        builder.add_op(Phase(phase), (c,))

    return builder.build()


def control_target_condition(op: AbstractOp) -> SubResolverCondition:
    base_id = op.base_id

    def cond(op_id: Ident) -> bool:
        assert len(op_id.params) > 0
        target_op = op_id.params[0]
        assert isinstance(target_op, Op)
        return target_op.base_id == base_id

    return cond


def controlled_x_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(CNOT, builder.qubits)
    return builder.build()


def controlled_y_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    c, t = builder.qubits
    builder.add_op(Sdag, (t,))
    builder.add_op(CNOT, (c, t))
    builder.add_op(S, (t,))
    return builder.build()


def controlled_z_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(CZ, builder.qubits)
    return builder.build()


def controlled_h_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    builder.add_op(RY(math.pi / 4), (q1,))
    builder.add_op(CZ, (q0, q1))
    builder.add_op(RY(-math.pi / 4), (q1,))
    return builder.build()


def _crx(builder: SubBuilder, q0: Qubit, q1: Qubit, angle: float) -> None:
    builder.add_op(H, (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(-angle / 2), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(angle / 2), (q1,))
    builder.add_op(H, (q1,))


def controlled_rx_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _crx(builder, q0, q1, angle)
    return builder.build()


def _cry(builder: SubBuilder, q0: Qubit, q1: Qubit, angle: float) -> None:
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RY(-angle / 2), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RY(angle / 2), (q1,))


def controlled_ry_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _cry(builder, q0, q1, angle)
    return builder.build()


def _crz(builder: SubBuilder, q0: Qubit, q1: Qubit, angle: float) -> None:
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(-angle / 2), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(angle / 2), (q1,))


def controlled_rz_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _crz(builder, q0, q1, angle)
    return builder.build()


def controlled_phase_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _crz(builder, q0, q1, angle)
    builder.add_op(Phase(angle / 2), (q0,))
    return builder.build()


def controlled_sqrtx_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _crx(builder, q0, q1, math.pi / 2)
    return builder.build()


def controlled_sqrtxdag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _crx(builder, q0, q1, -math.pi / 2)
    return builder.build()


def controlled_sqrty_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _cry(builder, q0, q1, math.pi / 2)
    return builder.build()


def controlled_sqrtydag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    _cry(builder, q0, q1, -math.pi / 2)
    return builder.build()


def controlled_s_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Tdag, (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(T, (q0,))
    builder.add_op(T, (q1,))
    return builder.build()


def controlled_sdag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(T, (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Tdag, (q0,))
    builder.add_op(Tdag, (q1,))
    return builder.build()


def controlled_t_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(-math.pi / 8), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(math.pi / 8), (q0,))
    builder.add_op(Phase(math.pi / 8), (q1,))
    return builder.build()


def controlled_tdag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(math.pi / 8), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(-math.pi / 8), (q0,))
    builder.add_op(Phase(-math.pi / 8), (q1,))
    return builder.build()


def controlled_cnot_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(Toffoli, builder.qubits)
    return builder.build()


def controlled_cz_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(MultiControlled(Z, 2, 0b11), builder.qubits)
    return builder.build()


def controlled_swap_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    q0, q1, q2 = builder.qubits
    builder.add_op(CNOT, (q2, q1))
    builder.add_op(Toffoli, (q0, q1, q2))
    builder.add_op(CNOT, (q2, q1))
    return builder.build()


def controlled_toffoli_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(MultiControlled(X, 3, 0b111), builder.qubits)
    return builder.build()


def controlled_multicontrolled_resolver(op: Op, repository: SubRepository) -> Sub:
    target_op = op.id.params[0]
    assert isinstance(target_op, Op)
    inner_op, control_bits, control_value = target_op.id.params
    assert isinstance(inner_op, Op)
    assert isinstance(control_bits, int)
    assert isinstance(control_value, int)
    control_bits += 1
    control_value = (control_value << 1) + 1
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(
        MultiControlled(inner_op, control_bits, control_value), builder.qubits
    )
    return builder.build()


_repo = default_repository()
_repo.register_sub_resolver(Controlled, controlled_sub_resolver)

_resolvers: Collection[tuple[AbstractOp, SubResolver]] = [
    (X, controlled_x_resolver),
    (Y, controlled_y_resolver),
    (Z, controlled_z_resolver),
    (H, controlled_h_resolver),
    (RX, controlled_rx_resolver),
    (RY, controlled_ry_resolver),
    (RZ, controlled_rz_resolver),
    (Phase, controlled_phase_resolver),
    (SqrtX, controlled_sqrtx_resolver),
    (SqrtXdag, controlled_sqrtxdag_resolver),
    (SqrtY, controlled_sqrty_resolver),
    (SqrtYdag, controlled_sqrtydag_resolver),
    (S, controlled_s_resolver),
    (Sdag, controlled_sdag_resolver),
    (T, controlled_t_resolver),
    (Tdag, controlled_tdag_resolver),
    (CNOT, controlled_cnot_resolver),
    (CZ, controlled_cz_resolver),
    (SWAP, controlled_swap_resolver),
    (Toffoli, controlled_toffoli_resolver),
    (MultiControlled, controlled_multicontrolled_resolver),
]

for target, resolver in _resolvers:
    _repo.register_sub_resolver(Controlled, resolver, control_target_condition(target))
