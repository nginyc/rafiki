import {
    takeLatest,
    call,
    put,
    fork,
    select
} from "redux-saga/effects"
import { showLoading, hideLoading } from 'react-redux-loading-bar'
import * as actions from "../containers/Models/actions"
import { notificationShow } from "../containers/Root/actions.js"
import * as api from "../services/ClientAPI"
import { getToken } from "./utils";

// Watch action request Model list and run generator getModelList
function* watchGetModelListRequest() {
    yield takeLatest(actions.Types.REQUEST_MODEL_LIST, getModelList)
}

/* for List Model command */
function* getModelList() {
    try {
        console.log("Start to load models")
        yield put(showLoading())
        const token = yield select(getToken)
        // TODO: implement API requestListModel
        const models = yield call(api.requestModelList, {}, token)
        console.log("Model loaded", models.data)
        yield put(actions.populateModelList(models.data))
        yield put(hideLoading())
    } catch (e) {
        console.error(e.response)
        console.error(e)
        yield put(notificationShow("Failed to Fetch Model List"));
        // TODO: implement notification for success and error of api actions
        // yield put(actions.getErrorStatus("failed to deleteUser"))
    }
}

// TO BE IMPLEMENTED 

// function* watchPostModelsRequest() {
//     yield takeLatest(actions.Types.CREATE_Model, createModel)
// }

// function* createModel(action) {
//     const {name, task, file, Model_url} = action
//     try {
//         const token = yield select(getToken)
//         yield call(api.postCreateModel, name, task, file, Model_url, token)
//         console.log("Create Model success")
//         yield put(notificationShow("Create ModelList Successfully")); // no need to write test for this 
//     } catch(e) {
//         console.error(e.response)
//         console.error(e)
//         console.error(e.response.data)
//         yield put(notificationShow("Failed to Create Dataset"));
//     }
// }

// fork is for process creation, run in separate processes
export default [
    fork(watchGetModelListRequest)
]

