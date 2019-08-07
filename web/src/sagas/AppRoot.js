import {
  takeLatest,
  call,
  fork,
  put,
  delay,
  all
} from "redux-saga/effects"
import * as actions from "../containers/Root/actions"
import * as api from "../services/AuthAPI"


export function* authLogin(action) {
  try{
    yield put(actions.authStart())
    const res = yield call(api.requestSignIn, action.authData)
    const user_id = res.data.user_id
    const token = res.data.token
    // 1 hour expirationTime?
    const expirationTime = 3600 * 1000
    const expirationDate = new Date(new Date().getTime() + expirationTime);
    localStorage.setItem('user_id', user_id);
    localStorage.setItem('token', token);
    localStorage.setItem('expirationDate', expirationDate);
    yield all([put(actions.notificationShow("Successfully signed in")),put(actions.authSuccess(token,user_id))])
  } catch(e) {
    console.error(e)
    yield put(actions.authFail(e))
    yield put(actions.notificationShow("Failed to sign in"));
  }
}

export function* watchSignInRequest() {
  yield takeLatest(actions.Types.SIGN_IN_REQUEST, authLogin)
}

function* checkAuthState(action) {
  try{
    const token = localStorage.getItem('token');
    const user_id = localStorage.getItem('user_id');
    if (!token) {
      console.log("token not found")
      yield put(actions.logout())
    } else {
      const expirationDate = new Date(localStorage.getItem('expirationDate'));
      if (expirationDate <= new Date()) {
        console.log("token expired")
        yield put(actions.logout())
      } else {
        console.log("token found")
        yield put(actions.authSuccess(token,user_id))
        // after expiration auto logout
        yield delay(expirationDate.getTime() - new Date().getTime())
        localStorage.removeItem('token');
        localStorage.removeItem('expirationDate');
        localStorage.removeItem('user_id');
        yield put(actions.logout())
      }
    }
  } catch(e) {
    console.error(e)
    yield put(actions.authFail(e))
  }
}

function* watchAuthStateRequest() {
  yield takeLatest(actions.Types.AUTH_CHECK_STATE, checkAuthState)
}

function* autoHideNotification() {
  yield delay(3000);
  console.log("Auto Hide the Notification area")
  yield put(actions.notificationHide());
}

function* watchNotifications() {
  yield takeLatest(actions.Types.NOTIFICATION_SHOW, autoHideNotification);
}

// fork is for process creation, run in separate processes
const AppRootSagas = [
  fork(watchSignInRequest),
  fork(watchAuthStateRequest),
  fork(watchNotifications),
]

export default AppRootSagas
