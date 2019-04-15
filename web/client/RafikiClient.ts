import 'whatwg-fetch';

export default class RafikiClient {
  _storage: Storage;
  _adminHost: string;
  _adminPort: number;
  _token?: string;
  _user: User

  /*
    Initializes the Client to connect to a running 
    Rafiki Admin instance that the Client connects to.
  */
  constructor(
    adminHost: string = 'localhost', 
    adminPort: number = 3000, 
    storage?: Storage
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
  async login(email: string, password: string) {
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
   * Train Jobs
   * ***************************************/

  /*
    Get details of a train job on Rafiki.
  */
  async getTrainJob(app: string, appVersion: number = -1) {
    const data = await this._get(`/train_jobs/${app}/${appVersion}`);
    const trainJob = this._toTrainJob(data);
    return trainJob;
  }

  /*
    Lists all train jobs associated to an user on Rafiki.
  */
  async getTrainJobsByUser(userId: string) {
    const data = await this._get('/train_jobs', {
      'user_id': userId
    });
    const trainJobs = (<any[]>data).map((x) => this._toTrainJob(x));
    return trainJobs;
  }

  /*
    Lists all trials of an train job with associated app & app version.
  */
  async getTrialsOfTrainJob(app: string, appVersion: number = -1) {
    const data = await this._get(`/train_jobs/${app}/${appVersion}/trials`);
    const trials = (<any[]>data).map((x) => this._toTrial(x));
    return trials;
  }

  /* ***************************************
   * Inference Jobs
   * ***************************************/

  /*
    Lists all inference jobs associated to an user on Rafiki.
  */

  async getInferenceJobsByUser(user_id: string) {
    const data = await this._get('/inference_jobs', {
      user_id
    });
    const inferenceJobs = <InferenceJob[]>data;
    return inferenceJobs;
  }

  /* ***************************************
   * Trials
   * ***************************************/

  
  /*
    Gets a trial.
  */
  async getTrial(trialId: string) {
    const data = await this._get(`/trials/${trialId}`);
    const trial = this._toTrial(data);
    return trial;
  }

 /*
    Gets the logs for a trial.
  */
  async getTrialLogs(trialId: string) {
    const data = <any>(await this._get(`/trials/${trialId}/logs`));

    // Parse date strings into Dates
    for (const metric of <any[]>Object.values(data.metrics)) {
      metric.time = this._toDate(metric.time);
    }
    for (const message of <any[]>Object.values(data.messages)) {
      message.time = this._toDate(message.time);
    }

    const logs = <TrialLogs>data;
    return logs;
  }

  /* ***************************************
   * Private
   * ***************************************/

  _toTrial(x: any) {
    // Convert dates
    if (x.datetime_started) {
      x.datetime_started = this._toDate(x.datetime_started);
    }

    if (x.datetime_stopped) {
      x.datetime_stopped = this._toDate(x.datetime_stopped);
    }

    return <Trial>x;
  }

  _toTrainJob(x: any) {
    // Convert dates
    if (x.datetime_started) {
      x.datetime_started = this._toDate(x.datetime_started);
    }

    if (x.datetime_stopped) {
      x.datetime_stopped = this._toDate(x.datetime_stopped);
    }

    return <TrainJob>x;
  }
  
   _toDate(dateString: string) {
    const timestamp = Date.parse(dateString);

    // Check invalid date
    if (isNaN(timestamp)) {
      return null;
    }

    return new Date(timestamp);
  }

  _initializeStorage(storage?: Storage) {
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
    const user = this._storage.getItem('user');

    if (!token || !user) {
      return;
    }

    this._token = JSON.parse(token);
    this._user = JSON.parse(user);
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

  async _post(urlPath: string, json: object = {}, params: { [s: string]: any } = {}) {
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

  async _get(urlPath: string, params: object = {}) {
    const url = this._makeUrl(urlPath, params);
    const headers =  this._getHeaders();
    const res = await fetch(url, {
      method: 'GET',
      headers
    });
    
    const data = await this._parseResponse(res);
    return data;
  }

  async _parseResponse(res: Response) {
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

  _makeUrl(urlPath: string, params: { [s: string]: any } = {}) {
    const query = Object.keys(params)
      .map(k => `${encodeURIComponent(k)}=${encodeURIComponent(params[k])}`)
      .join('&');
    const queryString = query ? `?${query}` : '';
    return `http://${this._adminHost}:${this._adminPort}${urlPath}${queryString}`;
  }
}

export interface User {
  id: string;
  user_type: string;
}

export interface TrainJob {
  id: string;
  app: string;
  app_version: string;
  budget: {
    [budget_type: string]: number
  };
  datetime_started: Date;
  datetime_stopped?: Date;
  status: string;
  task: string;
}

export interface Trial {
  id: string;
  no: number;
  status: TrialStatus;
  datetime_started: Date;
  datetime_stopped?: Date;
  score: number;
  model_name: string;
  worker_id: string;
  proposal: object
}

export interface TrialLogs {
  plots: TrialPlot[];
  metrics: TrialMetric[];
  messages: TrialMessage[];
}

export interface TrialPlot {
  title: string;
  metrics: string[];
  x_axis: string;
}

export interface TrialMetric {
  time: Date;
  [metric: string]: any;
}

export interface TrialMessage {
  time?: Date;
  message: string;
}

export interface InferenceJob {
  id: string;
}

export type TrialStatus = 'STARTED' | 'RUNNING' | 'ERRORED' | 'TERMINATED' | 'COMPLETED';
