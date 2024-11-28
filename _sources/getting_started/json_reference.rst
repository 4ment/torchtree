Input files
===========

JSON file format
----------------

JSON is a human-readable format for storing and transmitting data. It is easy for humans to read and write and even easier for machines to parse and generate.
torchtree uses JSON files to describe models, such as phylogenetic models and parametric distributions, and algorithms.

The JSON format is built on two structures:

* A collection of key/value pairs enclosed in curly brackets which we refer to as JSON objects. 
* An ordered list of values enclosed in square brackets.

In both structures, values can be strings, numbers (integers or real numbers), booleans, lists, or other JSON objects.
The keys of JSON bjects are always strings.

An example of JSON file is shown below:

.. code-block:: JSON

    {
        "id": "gtr_model",
        "type": "torchtree.evolution.substitution_model.nucleotide.GTR",
        "rates": {
            "id": "gtr_rates",
            "type": "torchtree.core.parameter.Parameter",
            "tensor": [0.166, 0.166, 0.167, 0.167, 0.167, 0.167]
        },
        "frequencies": {
            "id": "gtr_frequencies",
            "type": "torchtree.core.parameter.Parameter",
            "tensor": [0.25, 0.25, 0.25, 0.25]
        }
    }

In this example the outermost JSON object contains 4 key/value pairs where the keys are :keycode:`id`, :keycode:`type`, :keycode:`rates` and :keycode:`frequencies`.
The values associated with :keycode:`id` and :keycode:`type` are strings, while the values associated with :keycode:`rates` and :keycode:`frequencies` are JSON objects.
The values associated with both :keycode:`tensor` keys are lists of real numbers.

Looking at the block of code it should be clear that the JSON object describes a GTR substitution model with identifier **gtr_model**.
We have also defined the relative rates and equilibrium state frequencies of the model as Parameter objects with identifiers **gtr_rates** and **gtr_frequencies** respectively.
Typically, this GTR object will be nested inside a TreeLikelihood object, along other objects such as tree model, site model, and alignment.


torchtree model specification
-----------------------------

Each JSON object corresponds to a Python object in torchtree, and the torchtree parser uses the value of the :keycode:`type` key to select the appropriate Python class for instantiating the corresponding object.
To ensure proper bookkeeping of objects, each JSON object must include an :keycode:`id` key associated with a unique identifier.
This allows multiple objects to reference the same instance of an object, which is common in hierarchical models.
This approach is similar to how BEAST uses XML files to create Java objects and it uses *id* and *idref* tags to reference objects.

Assuming the JSON GTR model is saved in a file called *gtr.json*, we can instantiate the model using the following code:

.. code-block:: python

    import json
    from torchtree.evolution.substitution_model.nucleotide import GTR

    with open('gtr.json', 'r') as file:
        model = json.load("gtr.json")
    stored_objects = {}
    gtr_model = GTR.from_json(model, stored_objects)

Each torchtree class that can be instantiated from a JSON object implements a class method called :py:meth:`~torchtree.core.serializable.JSONSerializable.from_json`.
In addition to instantiating the object, this class method is also responsible for instantiating the submodels and parameters required by the object.
If a submodel or parameter was already parsed and instantiated, this method will use the existing object that was stored in the ``stored_object`` dictionary.
Otherwise, it will create a new one and store it in the dictionary for future use, such as placing a prior on a parameter.

Naturally, the GTR model can also be instantiated directly in Python without turning to JSON:

.. code-block:: python

    import torch
    from torchtree import Parameter
    from torchtree.evolution.substitution_model.nucleotide import GTR

    gtr_rates = Parameter("gtr_rates", torch.tensor([0.166, 0.166, 0.167, 0.167, 0.167, 0.167]))
    gtr_frequencies = Parameter("gtr_frequencies", torch.tensor([0.25, 0.25, 0.25, 0.25]))

    gtr_model = GTR("gtr_model", gtr_rates, gtr_frequencies)


Some keywords are reserved for torchtree model specification. These are:

* :keycode:`id`: a unique identifier for the object/model. It should be unique across the whole file.
* :keycode:`type`: the name of the python class to instantiate an object. It should contain the class name and the full path to the class separated by dots.

Some rules for torchtree model specification are:

* Every object should have exactly one :keycode:`id` key and one :keycode:`type` key.
* torchtree JSON specification is case-sensitive.
  
  * For consistency keys should be written in lowercase (e.g. :keycode:`type` not :keycode:`Type`).
* If a key starts with an underscore, the key/value pair will be ignored by torchtree. Think of it as a way to comment things out.
* The root structure is a list and the parser will iterate over the list and instantiate the objects in the order they appear.

Objects that have been defined can be referenced by other objects. For example, the "rates" object can be referenced using its identifier **gtr_rates** when we define a Dirichlet prior on the rates.

.. code-block:: JSON

    {
        "id": "gtr_prior",
        "type": "torchtree.distributions.Distribution",
        "distribution": "torch.distributions.Dirichlet",
        "x": "gtr_rates",
        "parameters": {
            "concentration": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    }

Above, we have specified that the rates of the GTR model follow a flat Dirichlet distribution.
Looking at the :keycode:`type` and :keycode:`distribution` keys we can see that the object is a torchtree object of type :py:class:`~torchtree.distributions.distributions.Distribution` that uses the `Dirichlet <https://pytorch.org/docs/stable/distributions.html#dirichlet>`_ distribution from the ``torch.distributions`` module.
The :keycode:`x` key is used to reference the rates object with identifier **gtr_rates** defined earlier.

The concentration parameter of the Dirichlet distribution is a list of 6 ones, since it is not wrapped in a Parameter object it will be considered as a constant.
If we wanted to estimate the concentration parameter, we would have to wrap it in a Parameter object.

In the spirit of keeping the files compact and more readable, some of the most common objects in torchtree do not require the full path to the class name.
For example instead of using ``torchtree.core.parameter.Parameter`` in the :keycode:`type` key, we can simply use ``Parameter``.

Since creating JSON files can be difficult, we provide ``torchtree-cli`` a command line tool to generate the JSON files.
The user can then modify the file to ajust models and parameters, such as prior distributions and hyperparameters.


