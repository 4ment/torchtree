torchtree.core.utils
====================

.. py:module:: torchtree.core.utils


Attributes
----------

.. autoapisummary::

   torchtree.core.utils.REGISTERED_CLASSES


Exceptions
----------

.. autoapisummary::

   torchtree.core.utils.JSONParseError
   torchtree.core.utils.AlternativeAttributeError


Classes
-------

.. autoapisummary::

   torchtree.core.utils.TensorEncoder
   torchtree.core.utils.TensorDecoder
   torchtree.core.utils.SignalHandler


Functions
---------

.. autoapisummary::

   torchtree.core.utils.as_tensor
   torchtree.core.utils.tensor_rand
   torchtree.core.utils.get_class
   torchtree.core.utils.process_objects
   torchtree.core.utils.process_object_with_key
   torchtree.core.utils.process_object
   torchtree.core.utils.validate
   torchtree.core.utils.remove_comments
   torchtree.core.utils.replace_wildcard_with_str
   torchtree.core.utils.replace_star_with_str
   torchtree.core.utils.expand_plates
   torchtree.core.utils.update_parameters
   torchtree.core.utils.print_graph
   torchtree.core.utils.string_to_list_index
   torchtree.core.utils.package_contents
   torchtree.core.utils.register_class


Module Contents
---------------

.. py:data:: REGISTERED_CLASSES

.. py:class:: TensorEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)

   Bases: :py:obj:`json.JSONEncoder`


   Extensible JSON <http://json.org> encoder for Python data structures.

   Supports the following objects and types by default:

   +-------------------+---------------+
   | Python            | JSON          |
   +===================+===============+
   | dict              | object        |
   +-------------------+---------------+
   | list, tuple       | array         |
   +-------------------+---------------+
   | str               | string        |
   +-------------------+---------------+
   | int, float        | number        |
   +-------------------+---------------+
   | True              | true          |
   +-------------------+---------------+
   | False             | false         |
   +-------------------+---------------+
   | None              | null          |
   +-------------------+---------------+

   To extend this to recognize other objects, subclass and implement a
   ``.default()`` method with another method that returns a serializable
   object for ``o`` if possible, otherwise it should call the superclass
   implementation (to raise ``TypeError``).



   .. py:method:: default(obj)

      Implement this method in a subclass such that it returns
      a serializable object for ``o``, or calls the base implementation
      (to raise a ``TypeError``).

      For example, to support arbitrary iterators, you could
      implement default like this::

          def default(self, o):
              try:
                  iterable = iter(o)
              except TypeError:
                  pass
              else:
                  return list(iterable)
              # Let the base class default method raise the TypeError
              return JSONEncoder.default(self, o)




.. py:class:: TensorDecoder(*args, **kwargs)

   Bases: :py:obj:`json.JSONDecoder`


   Simple JSON <http://json.org> decoder

   Performs the following translations in decoding by default:

   +---------------+-------------------+
   | JSON          | Python            |
   +===============+===================+
   | object        | dict              |
   +---------------+-------------------+
   | array         | list              |
   +---------------+-------------------+
   | string        | str               |
   +---------------+-------------------+
   | number (int)  | int               |
   +---------------+-------------------+
   | number (real) | float             |
   +---------------+-------------------+
   | true          | True              |
   +---------------+-------------------+
   | false         | False             |
   +---------------+-------------------+
   | null          | None              |
   +---------------+-------------------+

   It also understands ``NaN``, ``Infinity``, and ``-Infinity`` as
   their corresponding ``float`` values, which is outside the JSON spec.



   .. py:method:: object_hook(dic)


.. py:function:: as_tensor(dct, dtype=torch.float64)

.. py:function:: tensor_rand(distribution, shape, dtype=None, device=None, requires_grad=False)

   Create a tensor with the given dtype and shape and initialize it using a
   distribution.

   Continuous distributions: normal, log_normal, uniform.
   Discrete distributions: random, bernoulli

   :param distribution: distribution as a string (e.g. 'normal(1.0,2.0)', 'normal',
    'normal()').
   :type distribution: str
   :param shape: shape of the tensor
   :type shape: Sequence[int]
   :param dtype: dtype of the tensor
   :type dtype: torch.dtype
   :param device: device of the tensor
   :type device: torch.device
   :return: tensor
   :rtype: torch.Tensor

   :example:
   >>> _ = torch.manual_seed(0)
   >>> t1 = tensor_rand('normal(1.0, 2.0)', (1,2), dtype=torch.float64)
   >>> t1
   tensor([[4.0820, 0.4131]], dtype=torch.float64)
   >>> _ = torch.manual_seed(0)
   >>> t2 = tensor_rand('normal(0.0, 1.0)', (1,2), dtype=torch.float64)
   >>> _ = torch.manual_seed(0)
   >>> t3 = tensor_rand('normal()', (1,2), dtype=torch.float64)
   >>> t2 == t3
   tensor([[True, True]])


.. py:function:: get_class(full_name: str) -> type

.. py:exception:: JSONParseError

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:function:: process_objects(data, dic, force_list=False, key=None)

.. py:function:: process_object_with_key(key, data, dic, default=None)

.. py:function:: process_object(data, dic)

.. py:class:: SignalHandler

   .. py:method:: exit(signum, frame)


.. py:function:: validate(data, rules)

.. py:function:: remove_comments(obj)

   Remove comments in dictionary representation of objects.

   - A key starting with an underscore results in the key/value pair to be removed.
   - A dictionary with key equal to *ignore* and value set to *True* results in its
     removal.


.. py:function:: replace_wildcard_with_str(obj, wildcard, value)

.. py:function:: replace_star_with_str(obj, value)

.. py:function:: expand_plates(obj, parent=None, idx=None)

.. py:function:: update_parameters(json_object, parameters) -> None

   Recursively replace tensor in json_object with tensors present in
   parameters.

   :param dict json_object: json object
   :param parameters: list of Parameters
   :type parameters: list(Parameter)


.. py:function:: print_graph(g: torch.Tensor, level: int = 0) -> None

   Print computation graph.

   :param torch.Tensor g: a tensor
   :param level: indentation level


.. py:exception:: AlternativeAttributeError

   Bases: :py:obj:`Exception`


   Custom exception for debugging conflicts between @property and
   __getattr__

   https://stackoverflow.com/questions/36575068/attributeerrors-undesired-interaction-between-property-and-getattr


   .. py:method:: wrapper(f)
      :classmethod:


      Wraps a function to reraise an AttributeError as the alternate
      type.



.. py:function:: string_to_list_index(index_str) -> Union[int, slice]

.. py:function:: package_contents(package_name)

.. py:function:: register_class(_cls, name=None)

