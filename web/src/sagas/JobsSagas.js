import { getToken } from "./utils"
import { takeLatest, put, call, select } from "redux-saga/effects";
import * as actions from "containers/Jobs/actions"
import { showLoading, hideLoading } from "react-redux-loading-bar";
import { notificationShow } from "containers/Root/actions";
import * as api from "services/ClientAPI"

export function* watchRequestJobsList() {
    yield takeLatest(actions.Types.REQUEST_TRAIN_JOBSLIST, getJobsList)
}

export function* getJobsList() {
    try {
        console.log("Getting jobs List")
        yield put(showLoading())
        const token = yield select(getToken)
        console.log(token)
        const TrainJobsList = yield call(api.requestTrainJobsList, {}, token)
        yield put(actions.updateJobsList(TrainJobsList.data))
        yield put(hideLoading())
    } catch (e) {
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("Failed to Fetch TrainJobsList"))
    }
}

export function* watchPostTrainJob() {
    yield takeLatest(actions.Types.POST_CREAT_TRAINJOB, postTrainJob)
}

export function* postTrainJob() {
    try {
        yield put(showLoading())
        const token = yield select(getToken)
        yield call(api.postTrainJob, json, token)
        yield put(hideLoading())
    } catch (e) {
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("fail to Create TrainJob"))
    }
}
