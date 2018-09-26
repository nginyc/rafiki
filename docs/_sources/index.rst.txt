.. Rafiki documentation master file, created by
   sphinx-quickstart on Mon Sep 17 16:39:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Rafiki's Documentation!
====================================================================

Index
--------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   
   docs/user/index.rst
   docs/dev/index.rst
   docs/classes/index.rst

What is Rafiki?
--------------------------------------------------------------------

*Rafiki* is a distributed, scalable system that trains machine learning (ML) models and deploys trained models, built with ease-of-use in mind.
To do so, it leverages on automated machine learning (AutoML).

For *app developers*, without any ML expertise, they can:

- Create a model training job for supported tasks, with their own datasets
- Deploy an ensemble of trained models for inference
- Integrate model predictions in their apps over HTTP

For *model developers*, they can:

- Contribute to Rafiki's pool of model templates

To use Rafiki, use the :ref:`rafiki-client` on the Python CLI.
