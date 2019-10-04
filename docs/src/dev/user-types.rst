.. _`user-types`:

Users of SingaAuto
====================================================================

.. figure:: ../images/system-context-diagram.jpg
    :align: center
    :width: 500px
    
    Users of SingaAuto

There are 4 types of users on SingaAuto:

    *Application Developers* create, manage, monitor and stop model training and serving jobs on SingaAuto. 
    They are the primary users of SingaAuto - they upload their datasets onto SingaAuto and create model training jobs that train on these datasets. 
    After model training, they trigger the deployment of these trained ML models as a web service that Application Users interact with. 
    While their model training and serving jobs are running, they administer these jobs and monitor their progress. 

    *Application Users* send queries to trained models exposed as a web service on SingaAuto, receiving predictions back. 
    Not to be confused with Application Developers, these users may be developers that are looking to conveniently integrate ML predictions into their mobile, web or desktop applications. 
    These application users have consumer-provider relationships with the aforementioned ML application developers, having delegated the work of training and deploying ML models to them.

    *Model Developers* create, update and delete model templates to form SingaAuto’s dynamic repository of ML model templates. 
    These users are key external contributors to SingaAuto, and represent the main source of up-to-date ML expertise on SingaAuto, 
    playing a crucial role in consistently expanding and diversifying SingaAuto’s underlying set of ML model templates for a variety of ML tasks. 
    Coupled with SingaAuto’s modern ML model tuning framework on SingaAuto, these contributions heavily dictate the ML performance that SingaAuto provides to Application Developers. 

    *SingaAuto Admins* create, update and remove users on SingaAuto. They regulate access of the other types of users to a running instance of SingaAuto.
