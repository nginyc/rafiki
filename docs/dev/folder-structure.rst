Rafiki's Folder Structure
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

    - `train_worker/`

        Code for Rafiki's `Train Worker` component

    - `inference_worker/`

        Code for Rafiki's `Inference Worker` component
    
    - `query_frontend/`

        Code for Rafiki's `Query Frontend` component

    - `db/`

        Code for an abstract access layer for Rafiki's *Database*

    - `cache/`

        Code for an abstract access layer for Rafiki's *Cache*

    - `model/`

        Code for the definition of abstract :class:`rafiki.model.BaseModel` that all models should extend, as well as utility methods for the implementation of models (e.g. reading from datasets).

    - `utils/`

        Collection of Rafiki-internal utility methods (e.g. for logging, serialization of models, authentication)

    - `config.py`

        Stores Rafiki-internal application-level configuration variables

    - `constants.py`

        Stores Rafiki's constants used internally & externally (e.g. valid values for user type, budget type, train job status)

- `dockerfiles/`
    
    Stores Dockerfiles for customized components of Rafiki 

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

