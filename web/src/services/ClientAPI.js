//axios to send ajax request
import axios from 'axios'
import HTTPconfig from "../HTTPconfig"

/* ========== Trials of Job ============*/

export const requestTrialsOfJob = (params, token, app, appVersion) => {
  const url = `/train_jobs/${app}/${appVersion}/trials`
  return _getWithToken(url, params, token)
}

/* ========== JobsList ============= */

/* Get a train job associated with an app & version */
export const getTrainJob = (params,token, app, appVersion) => {
  const url = `/train_jobs/${app}/${appVersion}`
  return _getWithToken(url, params, token)
}

export const requestTrainJobsList = (params, token) => {
  return _getWithToken("/train_jobs", params, token)
}

/* Create a training job*/
export const postCreateTrainJob = (json, token) => {
  const url = `/train_jobs`
  return _postJsonWithToken(url, json, token) 
}

/* ========== Dataset ============= */

export const requestDatasetList = (params, token) => {
  // require bearer token to do the authentication
  return _getWithToken("/datasets", params, token)
}

export const postCreateDataset = (name, task, file, dataset_url, token) => {
  // This function returns an Axios Promise object
  console.log("arguments", name, task, file, dataset_url)
  const formData = new FormData();
  if (file !== undefined) {
    console.log("submiting file")
    formData.append('dataset', file)
  } else {
    console.log("submiting url")
    formData.append("dataset_url", dataset_url)
  }
  formData.append("name", name)
  formData.append("task", task)
  console.log("dataset_url", formData.get("dataset_url"))
  return _postFormWithToken('/datasets', formData, token)
}

/* ========== Models ============= */
export const requestModelList = (params, token) => {
   // http://rafiki.nus:3000/models/available
   return _getWithToken("/models/available", params, token)
}

/* ========== Application(Inference Jobs) ============= */

// data = self._get('/inference_jobs', params={ 
  // 'user_id': user_id
// })

export const getInferenceJob = (params, token) => {
  return _getWithToken("/inference_jobs", params, token)
}

export const get_running_inference_jobs = (app, appVersion, params={}, token) => {
  return _getWithToken(`/inference_jobs/${app}/${appVersion}`, params, token)
}

export const createInferenceJob = (app, appVersion, budget, token) => {
  return _postJsonWithToken("/inference_jobs", {app, "app_version": appVersion, budget}, token)
}

// Private
export function _makeUrl(urlPath, params = {}) {
  const query = Object.keys(params)
    .map(k => `${encodeURIComponent(k)}=${encodeURIComponent(params[k])}`)
    .join('&');
  const queryString = query ? `?${query}` : '';
  const baseUrl = HTTPconfig.gateway
  const url = new URL(`${urlPath}${queryString}`, baseUrl)
  return url.toString()
}

export function _getHeader(token) {
  if (token) {
    return {
      "Authorization": `Bearer ${token}`
    }
  } else {
    return {}
  }
}

export const _getWithToken = (url, params, token) => {
  return axios({ // Axios(config) is a promise
    method: 'get',
    url: _makeUrl(url, params), // Use _makeUrl function to get the url
    headers: _getHeader(token)
  });
}

export const _postFormWithToken = (url, formData, token, params = {}) => {
  return axios({ // Axios(config) is a promise
    method: 'post',
    url: _makeUrl(url, params), // Use _makeUrl function to make the url
    headers: {
      "Authorization": `Bearer ${token}`
    },
    data: formData
  });
}

export const _postJsonWithToken = (url, json, token, params = {}) => {
  return axios({
    method: 'post',
    url: _makeUrl(url, params),
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    },
    data: json
  })
}