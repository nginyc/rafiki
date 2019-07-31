import { all } from 'redux-saga/effects';
import AppRootSagas from "./AppRoot"
import DatasetsSagas from "./DatasetsSagas"

export default function* rootSaga() {
  // similar to promise resolve all
  yield all([
    ...AppRootSagas,
    ...DatasetsSagas
  ]);
}
