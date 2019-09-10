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

# Loads the "Titantic" CSV dataset from `https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html` for the `TABULAR_REGRESSION` task
def load_titanic():
    load(
        dataset_url='https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
        out_train_dataset_path='data/titanic_train.csv',
        out_val_dataset_path='data/titanic_val.csv'
    )


if __name__ == '__main__':
    load_titanic()    
    