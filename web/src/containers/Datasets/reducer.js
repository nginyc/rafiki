import { Types } from "./actions"


const StatesToReset = {
}

const initialState = {
  // Ls-ds
  DatasetList: [ ],
  ...StatesToReset
};

export const DatasetsReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.POPULATE_DS_LIST:
      return {
        ...state,
        DatasetList: action.DatasetList.length === 0
          ? []
          : action.DatasetList
      }
    default:
      return state
  }
}

export default DatasetsReducer