# Preview notes

This is a preview of an RL-based simplifier for SymPy. It is modeled as an agent that selects from a set of actions.

First, a high-level choice is made among 4-options:

* Internal action
* External action
* "Teleport"
* Finish

Internal actions involve applying neural networks -- especially tree LSTMs and ordinary LSTMs -- to inspect the mathematical expression.

"Teleport" means selecting a sub-expression to "focus" on.

External actions are actual operations on the expression, such as commuting the order of summands / factors, distributing products, API calls to SymPy, defining new variables, saving the current expression as a checkpoint, etc.

The model is currently able to match, on average, sympy.simplify and sympy.factor, but not to exceed their performance. Further work is needed.

train/train.py is a script for initializing and training the model. The resulting training data can be analyzed using train/train_data_analysis.py.

The dataset was generated using random expression generators and "complicators".

# Links

Papers:
* [NeuRewriter](https://arxiv.org/pdf/1810.00337)
  * [Repo](https://github.com/facebookresearch/neural-rewriter/tree/master) 
* [Tree LSTM paper](https://aclanthology.org/P15-1150/) - "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
* [Deep learning for symbolic mathematics](https://arxiv.org/pdf/1912.01412)
* [Can neural networks learn symbolic rewriting?](https://arxiv.org/pdf/1911.04873)

Resources:
* Sympy
  * [Advanced expression manipulation](https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html)

