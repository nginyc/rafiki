#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from examples.datasets.tabular.csv_file import load

# Loads the "Body Fat" CSV dataset from `http://course1.winona.edu/bdeppa/Stat%20425/Datasets.html` for the `TABULAR_CLASSIFICATION` task
def load_body_fat():
    load(
        dataset_url='https://course1.winona.edu/bdeppa/Stat%20425/Data/bodyfat.csv',
        out_train_dataset_path='data/bodyfat_train.csv',
        out_val_dataset_path='data/bodyfat_val.csv'
    )
    
if __name__ == '__main__':
    load_body_fat()    
    