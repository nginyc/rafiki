import { combineReducers } from 'redux';

import { loadingBarReducer } from 'react-redux-loading-bar'

import { Root } from "../containers/Root/reducer"
import { ConsoleAppFrame } from "../containers/ConsoleAppFrame/reducer"
import { DatasetsReducer } from "../containers/Datasets/reducer"
import { JobsReducer } from "containers/Jobs/reducer"

import { connectRouter } from 'connected-react-router'



const rootReducer = (history) => {
  return combineReducers({
    loadingBar: loadingBarReducer,
    router: connectRouter(history),
    // app reducers:
    Root,
    ConsoleAppFrame,
    DatasetsReducer,
    JobsReducer,
  })
}

export default rootReducer;