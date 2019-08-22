/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import 'whatwg-fetch';

class RafikiClient {
  _storage;
  _adminHost;
  _adminPort;
  _token;
  _user

  /*
    Initializes the Client to connect to a running 
    Rafiki Admin instance that the Client connects to.
  */
  constructor(
    adminHost = 'localhost', 
    adminPort = 3000, 
    storage?
  ) {
    this._storage = this._initializeStorage(storage);
    this._adminHost = adminHost;
    this._adminPort = adminPort;
    this._token = null;
    this._user = null;

    // Try load saved user's login
    this._tryLoadUser();
  }

  /*
    Creates a login session as a Rafiki user. You will have to be logged in to perform any actions.
  */
  async login(email, password) {
    const data = await this._post('/tokens', {
      email, password
    });

    this._token = data.token;
    this._user = { 
      id: data.user_id, 
      'user_type': data.user_type 
    };

    // Persist user's login
    this._saveUser();

    return this._user;
  }

  /*
    Gets currently logged in user's data as `{ id, user_type }`, or `null` if client is not logged in.
  */
  getCurrentUser() {
    return this._user;
  }

  /*
    Clears the current login session.
  */
  logout() {
    this._token = null;
    this._user = null;
    this._clearUser();
  }

  /* ***************************************
   * Datasets
   * ***************************************/

  /*
    Create datasets from local file
  */
  async createDataset(name,task,file,dataset_url) {
    const formData = new FormData();
      if (file !== undefined) {
        formData.append('dataset',file)
      } else {
        formData.append("dataset_url", dataset_url)
      }
    formData.append("name", name)
    formData.append("task", task)
    const dataset = await this._postForm('/datasets', formData)
    return dataset 
  }

  async getDatasets(task) {
    let datasets = []
    if (task !== undefined) {
      datasets = await this._get('/datasets', task)
    } else {
      datasets = await this._get('/datasets')
    }
    return datasets
  }

   /* ***************************************
   * Models
   * ***************************************/

  async getAvailableModels(task) {
    const models = await this._get('/models/available', {
      task
    });
    return models
  }

  /* ***************************************
   * Train Jobs
   * ***************************************/

  /*
    Create a train jobs associated to an user on Rafiki.
  */
  async createTrainJob(json) {
    const data = await this._post('/train_jobs', json)
    const trainJob = data
    return trainJob;
  }

  /*
    Lists all train jobs associated to an user on Rafiki.
  */
  async getTrainJobsByUser(userId) {
    const data = await this._get('/train_jobs', {
      'user_id': userId
    });
    const trainJobs = data.map((x) => this._toTrainJob(x));
    return trainJobs;
  }

  /*
    Get a train job associated with an app & version
  */
  async getTrainJob(app, appVersion = -1) {
    const data = await this._get(`/train_jobs/${app}/${appVersion}`);
    const trainJob = data;
    return trainJob;
  }

  /*
    Lists all trials of an train job with associated app & app version.
  */
  async getTrialsOfTrainJob(app, appVersion = -1) {
    const data = await this._get(`/train_jobs/${app}/${appVersion}/trials`);
    const trials = (data).map((x) => this._toTrial(x));
    return trials;
  }

  /* ***************************************
   * Inference Jobs
   * ***************************************/

  /*
    Lists all inference jobs associated to an user on Rafiki.
  */

  async getInferenceJobsByUser(user_id) {
    const data = await this._get('/inference_jobs', {
      user_id
    });
    const inferenceJobs = data;
    return inferenceJobs;
  }

  /* ***************************************
   * Trials
   * ***************************************/

  
  /*
    Gets a trial.
  */
  async getTrial(trialId) {
    const data = await this._get(`/trials/${trialId}`);
    const trial = this._toTrial(data);
    return trial;
  }

 /*
    Gets the logs for a trial.
  */
  async getTrialLogs(trialId) {
    const data = await this._get(`/trials/${trialId}/logs`);

    // Parse date strings into Dates
    for (const metric of Object.values(data.metrics)) {
      metric.time = this._toDate(metric.time);
    }
    for (const message of Object.values(data.messages)) {
      message.time = this._toDate(message.time);
    }

    const logs = data;
    return logs;
  }

  /* ***************************************
   * Private
   * ***************************************/

  _toTrial(x) {
    // Convert dates
    if (x.datetime_started) {
      x.datetime_started = this._toDate(x.datetime_started);
    }

    if (x.datetime_stopped) {
      x.datetime_stopped = this._toDate(x.datetime_stopped);
    }

    return x;
  }

  _toTrainJob(x) {
    // Convert dates
    if (x.datetime_started) {
      x.datetime_started = this._toDate(x.datetime_started);
    }

    if (x.datetime_stopped) {
      x.datetime_stopped = this._toDate(x.datetime_stopped);
    }

    return x;
  }
  
   _toDate(dateString) {
    const timestamp = Date.parse(dateString);
    if (isNaN(timestamp)) {
      return null;
    }

    return new Date(timestamp);
  }

  _initializeStorage(storage) {
    if (storage) {
      return storage;
    } else if (window && window.localStorage) {
      return window.localStorage;
    } else if (window && window.sessionStorage) {
      return window.sessionStorage;
    }
  }

  _tryLoadUser() {
    if (!this._storage) {
      return;
    }

    const token = this._storage.getItem('token');
    const user_id = this._storage.getItem('user_id');

    if (!token || !user_id) {
      return;
    }

    this._token = token;
    this._user = { "user": { user_id } }
  }

  _saveUser() {
    if (!this._storage) {
      return;
    }
    
    // Persist token & user in storage
    this._storage.setItem('token', JSON.stringify(this._token));
    this._storage.setItem('user', JSON.stringify(this._user));
  }

  _clearUser() {
    if (!this._storage) {
      return;
    }

    this._storage.removeItem('token');
    this._storage.removeItem('user');
  }

  async _post(urlPath, json = {}, params = {}) {
    const url = this._makeUrl(urlPath, params);
    const headers =  this._getHeaders();
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(json)
    });
    
    const data = await this._parseResponse(res);
    return data;
  }

  /* this function is for form/file upload */
  async _postForm(urlPath, formData, params = {}) {
    const url = this._makeUrl(urlPath, params);
    const headers =  this._getHeaders();
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        ...headers,
      },
      body: formData
    });

    const responseData = await this._parseResponse(res);
    return responseData;
  }

  async _get(urlPath, params = {}) {
    const url = this._makeUrl(urlPath, params);
    const headers =  this._getHeaders();
    const res = await fetch(url, {
      method: 'GET',
      headers
    });
    
    const data = await this._parseResponse(res);
    return data;
  }

  async _parseResponse(res) {
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text);
    } 

    const data = await res.json();
    return data;
  }

  _getHeaders() {
    if (this._token) {
      return {
        'Authorization': `Bearer ${this._token}`
      };
    } else {
      return {};
    }
  }

  _makeUrl(urlPath, params = {}) {
    const query = Object.keys(params)
      .map(k => `${encodeURIComponent(k)}=${encodeURIComponent(params[k])}`)
      .join('&');
    const queryString = query ? `?${query}` : '';
    return `http://${this._adminHost}:${this._adminPort}${urlPath}${queryString}`;
  }
}

export default RafikiClient;