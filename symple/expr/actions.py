from collections import OrderedDict

import symple.expr.actions_field as af

ACTIONS = OrderedDict([
    (af.commute, af.can_commute),
    (af.associate_b, af.can_associate_b),
    (af.distribute_b, af.can_distribute_b),
    (af.undistribute_b, af.can_undistribute_b),
    (af.reduce_unit, af.can_reduce_unit),
    (af.multiply_unit, af.can_multiply_unit),
    (af.add_unit, af.can_add_unit),
    (af.cancel, af.can_cancel),
])