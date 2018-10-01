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

    - `train_worker/`

        Code for Rafiki's `Train Worker` component

    - `inference_worker/`

        Code for Rafiki's `Inference Worker` component
    
    - `predictor/`

        Code for Rafiki's `Predictor` component

    - `db/`

        Code for Rafiki's *Database* as an abstract data access layer

    - `cache/`

        Code for Rafiki's *Cache* as an abstract data access layer

    - `model/`

        Code for the definition of abstract :class:`rafiki.model.BaseModel` that all models should extend, as well as utility methods for the implementation of models (e.g. reading from datasets).

    - `containers/`

        Code for the deployment of Rafiki's dynamic stack (e.g. workers) as *services*

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

