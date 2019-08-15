import { getToken, getUserId } from "./utils"
import { fork, takeLatest, put, call, select } from "redux-saga/effects";
import * as actions from "containers/Jobs/actions"
import { showLoading, hideLoading } from "react-redux-loading-bar";
import { notificationShow } from "containers/Root/actions";
import { push } from 'connected-react-router'
import * as api from "services/ClientAPI"

/* ======= RequestJobsList =========*/

export function* watchRequestJobsList() {
    yield takeLatest(actions.Types.REQUEST_TRAIN_JOBSLIST, getJobsList)
}

export function* getJobsList(action) {
    try {
        // console.log("Getting jobs List")
        yield put(showLoading())
        const token = yield select(getToken)
        const user_id = yield select(getUserId)
        const TrainJobsList = yield call(api.requestTrainJobsList, {user_id}, token)
        yield put(actions.populateJobsList(TrainJobsList.data))
        yield put(hideLoading())
    } catch (e) {
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("Failed to Fetch TrainJobsList"))
    }
}

/* ===== CREATE TRAIN JOBS ====== */

export function* watchPostTrainJob() {
     yield takeLatest(actions.Types.POST_CREAT_TRAINJOB, postTrainJob)
}

export function* postTrainJob(action) {
    try {
        yield put(showLoading())
        const token = yield select(getToken)
        yield call(api.postCreateTrainJob, action.json, token)
        yield put(notificationShow("Create TrainJob success"))
        yield put(hideLoading())
        yield put(push('/console/jobs/list-train-jobs'))
    } catch (e) {
        yield call(alert, e.response.data)
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("fail to Create TrainJob"))
    }
}

/* ===== STOP TRAIN JOBS ====== */

// export function* watchRequestStopTrainjob() {
//     try {
//         yield put(showLoading())
//     }
// }

/*===== TRIALS =====*/

export function* watchRequestTrialsListOfJob() {
    yield takeLatest(actions.Types.REQUEST_TRIALSLIST_OFJOB, getTrialsListOfJob)
}

export function* getTrialsListOfJob(action) {
    try {
        // Get Trials list and display notification 
        yield put(showLoading()) 
        const token = yield select(getToken)
        yield call(getJobsList, action)
        console.log("Getting Trials List of Job")
        const trialsList = yield call(api.requestTrialsOfJob, {}, token, action.app, action.appVersion) 
        yield put(actions.populateTrialsToJobs(trialsList.data, action.app, action.appVersion))
        yield put(hideLoading())
    } catch(e) {
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("Failed to Fetch TrialsList"))
    }
}


export default [
    fork(watchRequestJobsList),
    fork(watchRequestTrialsListOfJob),
    fork(watchPostTrainJob)
]
