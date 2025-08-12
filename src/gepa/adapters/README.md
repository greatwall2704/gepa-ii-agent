# GEPA Adapters

> GEPA 🤝 Any Framework

This directory provides the interface to allow GEPA to plug into systems and frameworks of your choice! GEPA can interface with any system consisting of text components, by implementing `GEPAAdapter` in [../core/adapter.py](../core/adapter.py).

Currently, [DSPy](https://dspy.ai/) is the only framework for which an adapter has been implemented and it is available at [https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt/gepa](https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt/gepa).

We aspire to integrate GEPA support in many other frameworks!