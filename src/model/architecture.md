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
      * $$ 
        \begin{align*} 
            i_t  = & \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t  = & \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t  = & \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t  = & \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t  = & f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t  = & o_t \odot \tanh(c_t) \\
        \end{align*} 
        $$
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

# Possible imporvements

1. global state information
2. self attention
3. "scaffolding" - extra graph edges - kind of like self attention
4. Composition of operations
5. Selection of nodes for special attention
6. Modify LSTM - traditional LSTM might not be totally appropriate
7. Include sympy operations (e.g. factor)
8. replace embedding with one-hot
9. Separate data from structure
10. Optimizations:
    * [TorchScript](https://medium.com/@hihuaweizhu/key-points-to-grasp-for-torchscript-beginners-c02cf94aaa50)
    * Torch compile


# Questions

1. What are precisely the benefits of Actor-Critic?
2. Monte-Carlo vs TD - Monte Carlo more stable (according to NeuRewriter guys)?
3. What is GRU about anyway?


# Policy gradient

The total return is (the ensemble average over states $s$ of -) (ignoring the time-penalty):
$$ 
\begin{align*}
& \sum_a p(w, s, a) Q(s, a) \\
= &  \sum_a p(w, s, a) \left( \underbrace{C(s)-C(a(s))}_{\text{reward}\, = \, \text{change in node count}} + 
\gamma \sum_{a'} p(w, a(s), a') Q(a(s), a') \right) \\
= &  \sum_{a_0, a_1, \ldots, a_n} p(w, s, a_0) !!!
\end{align*}
$$
where $w$ are the weights of the neural net. If we take the derivative with respect to $w$, we get:
$$
\begin{align*}
\nabla_w \sum_a p(w, s, a) Q(s, a)  = & 
 \sum_a \nabla_w p(w, s, a) \left( C(s)-C(a(s)) + \gamma \sum_{a'} p(w, a(s), a') Q(a(s), a') \right) \\
+ &  \sum_a  p(w, s, a) \left(\gamma \nabla_w \sum_{a'} p(w, a(s), a') Q(a(s), a') \right) \\
= & \sum_a p(w, s, a)\nabla_w \log \left( p(w, s, a) \right)  Q(s, a) \\
+ &  \sum_a  p(w, s, a) \left(\gamma \nabla_w \sum_{a'} p(w, a(s), a') Q(a(s), a') \right) \\
\end{align*}
$$