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

    - `worker/`

        Code for Rafiki's `Train Worker` component and `Inference Worker` component
    
    - `predictor/`

        Code for Rafiki's `Predictor` component

    - `db/`

        Code for an abstract access layer for Rafiki's *Database*

    - `cache/`

        Code for an abstract access layer for Rafiki's *Cache*

    - `utils/`

        Collection of Rafiki-internal utility methods (e.g. for logging, authentication)

    - `config.py`

        Stores Rafiki-internal application-level configuration variables

    - `constants.py`

        Stores Rafiki's constants used internally & externally (e.g. valid values for user type, budget type, train job status)

    - `model.py`

        Stores definition of abstract :class:`rafiki.model.BaseModel` that all Rafiki models should extend, as well as Rafiki-internal methods for managing models.

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

