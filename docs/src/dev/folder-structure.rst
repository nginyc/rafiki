.. _`folder-structure`:

Folder Structure
====================================================================

- `singaauto/`

    SingaAuto's Python package 

    - `admin/`

        SingaAuto's static Admin component

    - `advisor/`

        SingaAuto's advisors

    - `client/`

        SingaAuto's client-side SDK

        .. seealso:: :class:`singaauto.client`

    - `worker/`

        SingaAuto's train, inference & advisor workers
    
    - `predictor/`

        SingaAuto's predictor

    - `meta_store/`

        Abstract data access layer for SingaAuto's main metadata store (backed by PostgreSQL)
    
    - `param_store/`

        Abstract data access layer for SingaAuto's store of model parameters (backed by filesystem)

    - `data_store/`

        Abstract data access layer for SingaAuto's store of datasets (backed by filesystem)

    - `cache/`

        Abstract data access layer for SingaAuto's temporary store of model parameters, train job metadata and queries & predictions in train & inference jobs (backed by Redis)

    - `container/`

        Abstract access layer for dynamic deployment of workers 

    - `utils/`

        Collection of SingaAuto-internal utility methods (e.g. for logging, authentication)

    - `model/`

        Definition of abstract :class:`singaauto.model.BaseModel` that all SingaAuto models should extend, programming 
        abstractions used in model development, as well as a collection of utility methods for model developers 
        in the implementation of their own models
    
    - `constants.py`

        SingaAuto's programming abstractions & constants (e.g. valid values for user types, job statuses)

- `web/`

    SingaAuto's Web Admin component
    
- `dockerfiles/`
    
    Stores Dockerfiles for customized components of SingaAuto 

- `examples/`
    
    Sample usage code for SingaAuto.

- `docs/`

    Source documentation for SingaAuto (e.g. Sphinx documentation files)

- `test/`

    Test code for SingaAuto

- `scripts/`

    Shell & python scripts for initializing, starting and stopping various components of SingaAuto's stack

- `.env.sh`

    Stores configuration variables for SingaAuto
