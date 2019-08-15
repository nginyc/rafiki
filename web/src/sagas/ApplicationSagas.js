import {
    takeLatest,
    call,
    put,
    fork,
    select
} from "redux-saga/effects"
import { showLoading, hideLoading } from 'react-redux-loading-bar'
import * as actions from "../containers/Application/actions"
import { notificationShow } from "../containers/Root/actions.js"
import * as api from "../services/ClientAPI"
import { getToken, getUserId } from "./utils";

// Watch action request Application list and run generator getApplicationList
function* watchGetApplicationListRequest() {
    yield takeLatest(actions.Types.FETCH_GET_INFERENCEJOB, getApplicationList)
}

/* for List Application command */
function* getApplicationList() {
    try {
        console.log("Start to load Applications")
        yield put(showLoading())
        const token = yield select(getToken)
        const user_id = yield select(getUserId)
        // TODO: implement API requestListApplication
        const Applications = yield call(api.getInferenceJob, {user_id}, token)
        console.log("Application loaded", Applications.data)
        yield put(actions.populateInferenceJob(Applications.data))
        yield put(hideLoading())
    } catch (e) {
        console.error(e.response)
        console.error(e)
        alert(e.response.data)
        yield put(notificationShow("Failed to Fetch Application List"));
        // TODO: implement notification for success and error of api actions
        // yield put(actions.getErrorStatus("failed to deleteUser"))
    }
}

// watch action call POST CREATE INFERENCEJOB => call apli.create inference job
function* watchCreateInferenceJobRequest() {
    yield takeLatest(actions.Types.POST_CREATE_INFERENCEJOB, createInferenceJob)
}

function* createInferenceJob(action) {
    const { app, appVersion, budget } = action
    try {
        const token = yield select(getToken)
        yield call(api.createInferenceJob, app, appVersion, budget, token)
        yield put(notificationShow("Create Inference Job Successfully")); // no need to write test for this 
    } catch(e) {
        console.error(e.response)
        console.error(e)
        alert(e.response.data)
        yield put(notificationShow("Failed to Create Inference Job"));
    }
}

// fork is for process creation, run in separate processes
export default [
    fork(watchGetApplicationListRequest),
    fork(watchCreateInferenceJobRequest)
]

