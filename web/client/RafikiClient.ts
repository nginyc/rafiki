import 'whatwg-fetch';

class RafikiClient {
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
    adminPort: number = 8000, 
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
    Lists all train jobs associated to an user on Rafiki.
  */
  async getTrainJobsByUser(userId: string) {
    const data = await this._get('/train_jobs', {
      'user_id': userId
    });
    const trainJobs = <TrainJob[]>data;
    return trainJobs;
  }

  async getTrialsOfTrainJob(app: string, appVersion: number = -1) {
    const data = await this._get(`/train_jobs/${app}/${appVersion}/trials`);
    const trials = <Trial[]>data;
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

  _toDate(dateString: string) {
    return new Date(Date.parse(dateString));
  }

  _initializeStorage(storage?: Storage) {
    if (storage) {
      return storage;
    } else if (window && window.sessionStorage) {
      return window.sessionStorage;
    } else if (window && window.localStorage) {
      return window.localStorage;
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

interface User {
  id: string;
  user_type: string;
}

interface TrainJob {
  id: string;
  app: string;
  app_version: string;
  budget_amount: number;
  datetime_started: string;
  datetime_completed: string;
  status: string;
  task: string;
}

interface Trial {
  id: string;
  status: string;
  datetime_started: string;
  datetime_stopped: string;
  score: number;
  model_name: string;
}

interface TrialLogs {
  plots: TrialPlot[];
  metrics: TrialMetric[];
  messages: TrialMessage[];
}

interface TrialPlot {
  title: string;
  metrics: string[];
  x_axis: string;
}

interface TrialMetric {
  time: Date;
  [metric: string]: any;
}

interface TrialMessage {
  time: Date;
  message: string;
}

interface InferenceJob {
  id: string;
}

export default RafikiClient;
export { User, TrainJob, InferenceJob, Trial, TrialLogs, TrialPlot };