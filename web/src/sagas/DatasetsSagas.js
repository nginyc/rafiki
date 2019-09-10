import {
    takeLatest,
    call,
    put,
    fork,
    select
} from "redux-saga/effects"
import { showLoading, hideLoading } from 'react-redux-loading-bar'
import * as actions from "../containers/Datasets/actions"
import { notificationShow } from "../containers/Root/actions.js"
import * as api from "../services/ClientAPI"
import { getToken } from "./utils";
import { push } from "connected-react-router"

// List Datasets
function* watchGetDSListRequest() {
    yield takeLatest(actions.Types.REQUEST_LS_DS, getDatasetList)
}
 
/* for List Dataset command */
function* getDatasetList() {
    try {
        yield put(showLoading())
        const token = yield select(getToken)
        const DSList = yield call(api.requestDatasetList, {}, token)
        console.log(DSList)
        yield put(actions.populateDSList(DSList.data))
        yield put(hideLoading())
    } catch (e) {
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("Failed to Fetch DatasetList"));
        // TODO: implement notification for success and error of api actions
        // yield put(actions.getErrorStatus("failed to deleteUser"))
    }
}

function* watchPostDatasetsRequest() {
    yield takeLatest(actions.Types.CREATE_DATASET, createDataset)
}

function* createDataset(action) {
    const {name, task, file, dataset_url} = action
    try {
        yield put(showLoading())
        const token = yield select(getToken)
        yield call(api.postCreateDataset, name, task, file, dataset_url, token)
        console.log("Create Dataset success")
        yield alert("Create Dataset success")
        yield put(notificationShow("Create Dataset Success")); // no need to write test for this 
        yield(push('console/datasets/list-dataset'))
        yield put(hideLoading())
    } catch(e) {
        console.error(e.response)
        console.error(e)
        console.error(e.response.data)
        yield put(notificationShow("Failed to Create Dataset"));
    }
}

// fork is for process creation, run in separate processes
export default [
                fork(watchGetDSListRequest),
                fork(watchPostDatasetsRequest)
               ]
