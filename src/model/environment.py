import torch

OPS = {
    "finish": lambda en: (en, 0, True),  # Finish
    "move up": lambda en: (en.p, 0, False),
    "move left": lambda en: (en.a, 0, False),
    "move right": lambda en: (en.b, 0, False),  # Navigate
    "pass": lambda en: (en, 0, False),
}  # Pass

OP_INDICES = {key: i for i, key in enumerate(OPS.keys())}
OPS = list(OPS.values())
NUM_OPS = len(OPS)


# Consider using Open-Ai Gym?
class Symple:
    """
    An RL environment with which the agent should interact. To enrich the s
    et of operations we must update the variable OPS and the method Symple.update_validity_mask. The operations should return (state : the new ExprNode, reward: float, don
    e : bool).
    """

    def __init__(
        self,
        en: "ExprNode",
        time_penalty: float = -0.02,
        node_count_importance_factor: float = 1.0,
        invalid_action_penalty: float = -10.0,
    ):
        self.expr = en
        self.state = en
        self.validity_mask = torch.ones(NUM_OPS, dtype=int)
        self.update_validity_mask()
        self.time_penalty = time_penalty
        self.node_count_importance_factor = node_count_importance_factor
        self.invalid_action_penalty = invalid_action_penalty

    def step(self, action: int) -> (float, bool):
        reward = self.time_penalty
        if not self.validity_mask[action]:
            reward += self.invalid_action_penalty
            return reward, False
        self.state, node_count_reduction, done = OPS[action](self.state)
        self.update_validity_mask()
        reward += self.node_count_importance_factor * node_count_reduction
        return reward, done

    def update_validity_mask(self):
        # Implement
        self.validity_mask[OP_INDICES["move up"]] = bool(self.state.p)
        self.validity_mask[OP_INDICES["move left"]] = bool(self.state.a)
        self.validity_mask[OP_INDICES["move right"]] = bool(self.state.b)
