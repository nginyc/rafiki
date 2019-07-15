# Rafiki

*Rafiki* is a distributed system that trains machine learning (ML) models and deploys trained models, built with ease-of-use in mind. To do so, it leverages on automated machine learning (AutoML).

Read Rafiki's full documentation at https://nginyc.github.io/rafiki/docs/latest

## Quick Setup

Prerequisites: MacOS or Linux environment

1. Install Docker 18

2. Install Python 3.6

3. Setup Rafiki's complete stack with the init script:

    ```sh
    bash scripts/start.sh
    ```

4. To destroy Rafiki's complete stack:

    ```sh
    bash scripts/stop.sh
    ```

More instructions are available in [Rafiki's Developer Guide](https://nginyc.github.io/rafiki/docs/latest/docs/src/dev).
## Data Prepare
put you data as the following format, wihch can be downloaded from http://cocodataset.org/  
```sh
\data  
    \coco  
        \annotations  
            \instances_train2014.json
            \instances_val2014.json
            \instances_minival2014.json
            \instances_valminusminival2014.json
        \train2014
            \images_train
        \val2014
            \images_val
        \test2014
            \images_test
```

## Issues

Report the issues at [JIRA](https://issues.apache.org/jira/browse/SINGA) or [Github](https://github.com/nginyc/rafiki/issues)


## Acknowledgements

The research is supported by the National Research Foundation, Prime Minister’s Office, Singapore under its National Cybersecurity R\&D Programme (Grant No. NRF2016NCR-NCR002-020), National Natural Science Foundation of China (No. 61832001), National Key Research and Development Program of China  (No. 2017YFB1201001), China Thousand Talents Program for Young Professionals (3070011 181811).
