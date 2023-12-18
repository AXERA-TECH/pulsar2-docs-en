=========================================
Accuracy Tuning Suggestions
=========================================

------------------------------
Accuracy loss troubleshooting
------------------------------

When accuracy loss occurs in the converted model, please follow the following recommended methods to troubleshoot the ``stage`` or ``layer`` where the problem occurs.

~~~~~~~~~~~~~~~~
CheckLists
~~~~~~~~~~~~~~~~

* The first step is to identify the hardware platform where accuracy loss occurs.

  * It only occurs on the ``AX`` platform, please continue to troubleshoot. ``Point drop problems occur on other platforms.`` It is a common problem. Users need to consider whether to train a better model and then re-quantize; 

* The second step is to determine the stage at which accuracy loss occurs.

  * ``pulsar2 build`` Low bisection accuracy (``Cosin Distance < 98%``)

    * ``Please follow the [Step 3] suggestions to continue troubleshooting.``

  * Connect the user's `post-processing` program to the board, and the accuracy after analysis is very low.

    * ``Please follow [Step 4] suggestions to continue troubleshooting.``

* The third step, ``Cosin Distance`` lower than 98%

  * Troubleshooting suggestions

    * Use layer-by-layer bisection to view the ``layers`` where accuracy loss occurs, and refer to the accuracy tuning recommendations for tuning.

* The fourth step is low board accuracy., 

  * Troubleshooting suggestions

    * Determine if post-processing is correct
    * If there is no problem with post-processing. Please contact ``AX`` to report any issues.

* Step fifth, seek help from AXera

  * When the user still cannot solve the problem through the first four steps of troubleshooting suggestions, please send the relevant ``log`` and ``conclusion`` to the ``FAE`` students so that ``AX`` engineers can locate the problem.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Layer-by-layer accuracy comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pulsar2 build`` Provides a set of layer-by-layer accuracy comparison tools between floating-point models and quantized models. For specific usage, please refer to :ref:`《Layer-by-Layer Bisection》 <perlayer_precision_debug_en>` chapter

~~~~~~~~~~~~~~~~~~~~~
Other things to note
~~~~~~~~~~~~~~~~~~~~~

If you need ``AX`` engineers to troubleshoot the problem, please provide detailed log information and relevant experimental conclusions.

.. note::

    If the minimum recurrence set can be provided, the efficiency of problem solving can be improved.

----------------------------
Accuracy Tuning Suggestions
----------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adjust quantitative data and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Adjust ``mean`` and ``std``: keep ``mean`` and ``std`` consistent with the preprocessing parameters of the model during training;
* Adjust ``rgb/bgr`` order: This parameter will also affect the quantization accuracy and needs to be consistent with the input during model training; it can be configured through the ``input_configs.tensor_format`` field in ``quant``;
* Quantitative data:
  
  * The calibration pictures should be as consistent as possible with the usage scenarios.
  * The number of samples in each category should be balanced and cover all categories

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adjust quantitative strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, three quantitative strategies are supported: ``MinMax``, ``MSE``, and ``Percentile``. This can be set by modifying the calibration_method in the quant field. suggestions below:

* First use the ``MinMax`` strategy for quantification. If the accuracy is not ideal, try other quantification strategies;
* For classification and detection models: It is recommended to use ``MinMax`` and ``Percentile`` for quantification;
* For sequence models: It is recommended to use the ``MSE`` strategy for quantification. 

~~~~~~~~~~~~~~~~~~
Mixed precision
~~~~~~~~~~~~~~~~~~

If the accuracy still does not meet the requirements after trying different quantization strategies, you can find the layer with a cosine similarity value lower than 98% through layer-by-layer accuracy analysis and comparison, and set the quantization accuracy of this layer to ``U16``.
For details, please refer to :ref:`《Detailed Explanation of Mixed Precision Quantization》 <mix_precision_quantization_en>`
