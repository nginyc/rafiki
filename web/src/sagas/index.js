import { all } from 'redux-saga/effects';
import AppRootSagas from "./AppRoot";
import DatasetsSagas from "./DatasetsSagas";
import JobsSagas from "./JobsSagas";
import ModelsSagas from './ModelsSagas';
import ApplicationSagas from "./ApplicationSagas";

export default function* rootSaga() {
  // similar to promise resolve all 
  yield all([ // remember to add "..." in front of the lists.
    ...AppRootSagas,
    ...DatasetsSagas,
    ...JobsSagas,
    ...ModelsSagas,
    ...ApplicationSagas
  ]);
}
