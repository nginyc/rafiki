# Rafiki

*Rafiki* is a distributed system that trains machine learning (ML) models and deploys trained models, built with ease-of-use in mind. To do so, it leverages on automated machine learning (AutoML).

Read Rafiki's full documentation at https://nginyc.github.io/rafiki/docs/latest

## Quick Setup

Prerequisites: MacOS or Linux environment

1. Install Docker 18 ([Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/), [MacOS](https://docs.docker.com/docker-for-mac/install/)) and, if required, add your user to `docker` group ([Linux](https://docs.docker.com/install/linux/linux-postinstall/>))

2. Install Python 3.6 ([Ubuntu](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/), [MacOS](https://www.python.org/downloads/mac-osx/))

3. Clone this project (e.g. with [Git](https://git-scm.com/downloads>))

4. Setup Rafiki's complete stack with the setup script:

    ```sh
    bash scripts/start.sh
    ```

To completely destroy Rafiki's stack:

```sh
bash scripts/stop.sh
```

More instructions are available in [Rafiki's Developer Guide](https://nginyc.github.io/rafiki/docs/latest/src/dev).


## Issues

Report any issues at [Apache SINGA's JIRA](https://issues.apache.org/jira/browse/SINGA) or [Rafiki's Github Issues](https://github.com/nginyc/rafiki/issues)


## Acknowledgements

The research is supported by the National Research Foundation, Prime Minister’s Office, Singapore under its National Cybersecurity R\&D Programme (Grant No. NRF2016NCR-NCR002-020), National Natural Science Foundation of China (No. 61832001), National Key Research and Development Program of China  (No. 2017YFB1201001), China Thousand Talents Program for Young Professionals (3070011 181811).
