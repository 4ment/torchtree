Building blocks
===============

Parameter object
----------------
:py:class:`~torchtree.core.parameter.Parameter` objects play a central role in torchtree. They are used to define the parameters of models and distributions, and are involved in any kind of optimization or inference.
A Parameter object contains a reference to a pytorch tensor object which can be accessed using the :py:attr:`~torchtree.core.parameter.Parameter.tensor` property.
There are different ways to define a Parameter object in a JSON file. The most common way is to define a :keycode:`tensor` key associated with a list of real numbers as shown below:

.. code-block:: JSON

    {
        "id": "gtr_frequencies",
        "type": "Parameter",
        "tensor": [0.25, 0.25, 0.25, 0.25]
    }

Inside torchtree, this JSON object will be converted to a python object:

.. code-block:: python

    Parameter("gtr_frequencies", torch.tensor([0.25, 0.25, 0.25, 0.25]))

Another way to define the same object using a different initialization method is:

.. code-block:: JSON

    {
        "id": "gtr_frequencies",
        "type": "Parameter",
        "full": [4],
        "value": 0.25
    }

which in python will be converted to:

.. code-block:: python
    :linenos:

    Parameter("gtr_frequencies", torch.full([4], 0.25))


TransformedParameter object
---------------------------
In torchtree, :py:class:`~torchtree.core.parameter.Parameter` objects are typically considered to be unconstrained.
Optimizers (such as those used in ADVI and MAP) and samplers (e.g. HMC) will change the value of the tensor they encapsulate without checking if the new value is within the parameter's domain.
However, in many cases, phylogenetic models contain constrained parameters such as branch lengths (positive real numbers) or equilibrium base frequencies (positive real numbers that sum to 1).
For example, the GTR model expects the equilibrium base frequencies to be positive real numbers that sum to 1 and a standard optimizer will ignore such constraints.
:py:class:`~torchtree.core.parameter.TransformedParameter` objects allow moving from unconstrained to constrained spaces using `transform <https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.Transform>`_ objects available in pytorch.

We can replace the JSON object defining the GTR equilibrium base frequencies with a TransformedParameter object as shown below:

.. code-block:: JSON

    {
        "id": "gtr_frequencies",
        "type": "TransformedParameter",
        "transform": "torch.distributions.StickBreakingTransform",
        "x":{
            "id": "gtr_frequencies_unconstrained",
            "type": "TransformedParameter",
            "type": "Parameter",
            "zeros": [3]
        }
    }

This is equivalent to the following python code:

.. code-block:: python

    import torch
    from torchtree import Parameter, TransformedParameter

    unconstrained = Parameter("gtr_frequencies_unconstrained", torch.zeros([3]))
    transform = torch.distributions.StickBreakingTransform()
    constrained = TransformedParameter("gtr_frequencies", unconstrained, transform)
    
An optimizer will change the value of the **gtr_frequencies_unconstrained** Parameter object and the **gtr_frequencies** (transformed) parameter will apply the StickBreakingTransform transform to the value of **gtr_frequencies_unconstrained** to update the transition rate matrix.

In this example, we are using the `StickBreakingTransform <https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StickBreakingTransform>`_ object that will transform the unconstrained parameter **gtr_frequencies_unconstrained** to a constrained parameter **gtr_frequencies**.
Note the value of the :keycode:`transform` key is a string containing the full path to the pytorch class that implements the transformation.
Specifically, ``torch`` is the package name, ``distributions`` is the module name, and ``StickBreakingTransform`` is the class name.


Models and CallableModels
-------------------------
Virtually every torchtree object that does some kind of computations inherits from the :py:class:`~torchtree.core.model.Model` class.
Computations can involve Parameter and/or other Model objects.
The Distribution class we described earlier is derived from the class Model since it defines a probability distribution and return a log probability.
The GTR substitution model is also a Model object since its role is to calculate a transition probability matrix.

A model that returns a value when called is said to be *callable* and it extends the :py:class:`~torchtree.core.model.CallableModel` abstract class.
A distribution is a callable model since it returns the log probability of a sample.
The class representing a tree likelihood model is also callable since it calculates the log likelihood and we will describe it further in the next section.

