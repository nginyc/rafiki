.. _`folder-structure`:

Folder Structure
====================================================================

- `rafiki/`

    Rafiki's Python Package 

    - `admin/`

        Code for Rafiki's `Admin` component

    - `advisor/`

        Code for Rafiki's `Advisor` component

    - `client/`

        Code for Rafiki's `Client` component

        .. seealso:: :class:`rafiki.client.Client`

    - `worker/`

        Code for Rafiki's `Train Worker` component and `Inference Worker` component
    
    - `predictor/`

        Code for Rafiki's `Predictor` component

    - `meta_store/`

        Code for Rafiki's *MetaStore* as an abstract data access layer
    
    - `param_store/`

        Code for Rafiki's *ParamStore* as an abstract data access layer

    - `cache/`

        Code for Rafiki's *Cache* as an abstract data access layer

    - `container/`

        Code for the deployment of Rafiki's dynamic stack (e.g. workers) as *services*

    - `utils/`

        Collection of Rafiki-internal utility methods (e.g. for logging, authentication)

    - `model/`

        Stores definition of abstract :class:`rafiki.model.BaseModel` that all Rafiki models should extend,
        as well as a collection of utility methods for model developers in the implementation of their own models.
    
    - `config.py`

        Stores Rafiki-internal application-level configuration variables

    - `constants.py`

        Stores Rafiki's constants used internally & externally (e.g. valid values for user type, budget type, train job status)

- `web/`

    Code for Rafiki's `Admin Web` component
    
- `dockerfiles/`
    
    Stores Dockerfiles for customized components of Rafiki 

- `examples/`
    
    Stores sample model definitions and usage of Rafiki.

- `docs/`

    Stores all source documentation for Rafiki (e.g. Sphinx documentation files)

- `scripts/`

    Stores shell & python scripts for initializing, starting and stopping various components of Rafiki's stack

- `.env.sh`

    Stores Rafiki-internal build & deployment configuration variables 

- `conf.py`

    Sphinx documentation configuration file
    
- `index.rst`

    Sphinx master documentation file

