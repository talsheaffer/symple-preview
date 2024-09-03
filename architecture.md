# Architecture - the simplifier - an agent that simplifies

An RL framework.

*   Environment - The space of expressions
*   Rewards - Reductions in node count
*   State - the expression

--------------------------------------------

Actor-Critic with a "model":

*   Actor == policy == probability(of action given (state , model)). The actor is optimized to maximize value as predicted by the critic.
*   Critic == value(state, model). The critic is optimized to predict total return.
*   Cool additional option: Module for estimating whether two expressions are identical.

--------------------------------------------

The model (note: perhaps "model" is a bad name here. The "world" is directly visible to the agent, so perhaps this should be called "hidden state" or "latent representation"):

* Embedding vectors for each node in the expression graph.
* The vectors get "contextualized" by their neighbors. This can be done either by:
    * LSTM: either [graphical](https://aclanthology.org/P15-1150/) or sequential. In one LSTM step each node-vector gets contextuallized by immediate neighbors. When repeated, information propagates all over the graph. Can be applied to entire expression or to sub-expression, or even to a single node.
    * Self-attention: As in transformer. Possibly better for long-distance relationships among distant parts of the expression. Probably more compute intensive.
* In addition to node-vectors, additional metadata:
    * The simplifier's "position" (a subexpression).
    * The total node-count.
    * Node count of current subexpression.
    * Number of "children" of current node.
    * ...


--------------------------------------------

Actions:

I imagine the simplifier as having a "position" - a node in the graph. Actions taken affect the corresponding subexpression.

The actions come in a few types:

*   Move: Change node. Affects neither the state nor node-vectors.
    * Move up
    * Move down
    * "Teleport"
    * Go to trunk
    * ...
*   Learn: Doesn't affect the state - only the node-vectors.
    * Apply LSTM to contextualize node vector/s.
    * ...
*   Act: Apply transformation to current subexpression. Affects subexpression. May affect graph topology. The subexpression might need to be re-embedded.
    * Atomic transformations (e.g. distribute).
    * Sympy transformations (e.g. factor, expand)
    * Substitutions...
    * ...
*   Store: ...


# Issues

1. Should the tree be reduced to binary + unary or kept general? Keeping general might be more complicated, but seems more natural.
2. Should we freeze the tree (e.g. using sympy's "unevaluated expression") to prevent unexpected topological transitions? The agent can always apply expr.doit() to a subexpression to unfreeze, but it might need to relearn the corresponding sub-expression, since all kinds of changes might happen there.
3. Do we need Tree or Tree-manager object classes, or do we work with the expressions directly?
4. LSTM pros and cons:

| pros    | cons |
| -------- | ------- |
| Adjacency is very significant in math expressions | Difficult to notice distant but recurring subexpression |
| Perhaps can be used to "focus" on a specific junction and repeatedly query its neighbors for more context | NeuRewriter did it |

5. Self-attention pros and cons: Basically complement the above

<!-- | pros | cons |
| -------- | ------- |
|  |  | -->

6. What are the atomic operations and how to combine them? 




