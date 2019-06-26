.. _`folder-structure`:

Folder Structure
====================================================================

- `rafiki/`

    Rafiki's Python package 

    - `admin/`

        Rafiki's static Admin component

    - `advisor/`

        Rafiki's advisors

    - `client/`

        Rafiki's client-side SDK

        .. seealso:: :class:`rafiki.client`

    - `worker/`

        Rafiki's train, inference & advisor workers
    
    - `predictor/`

        Rafiki's predictor

    - `meta_store/`

        Abstract data access layer for Rafiki's main metadata store (backed by PostgreSQL)
    
    - `param_store/`

        Abstract data access layer for Rafiki's store of model parameters (backed by filesystem)

    - `data_store/`

        Abstract data access layer for Rafiki's store of datasets (backed by filesystem)

    - `cache/`

        Abstract data access layer for Rafiki's temporary store of model parameters, train job metadata and queries & predictions in train & inference jobs (backed by Redis)

    - `container/`

        Abstract access layer for dynamic deployment of workers 

    - `utils/`

        Collection of Rafiki-internal utility methods (e.g. for logging, authentication)

    - `model/`

        Definition of abstract :class:`rafiki.model.BaseModel` that all Rafiki models should extend, programming 
        abstractions used in model development, as well as a collection of utility methods for model developers 
        in the implementation of their own models
    
    - `constants.py`

        Rafiki's programming abstractions & constants (e.g. valid values for user types, job statuses)

- `web/`

    Rafiki's Web Admin component
    
- `dockerfiles/`
    
    Stores Dockerfiles for customized components of Rafiki 

- `examples/`
    
    Sample usage code for Rafiki.

- `docs/`

    Source documentation for Rafiki (e.g. Sphinx documentation files)

- `test/`

    Test code for Rafiki

- `scripts/`

    Shell & python scripts for initializing, starting and stopping various components of Rafiki's stack

- `.env.sh`

    Stores configuration variables for Rafiki
