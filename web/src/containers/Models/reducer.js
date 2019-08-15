import { Types } from "./actions"

const StatesToReset = {
}

const initialState = {
  // MODEL-List
  ModelList: [],
  ...StatesToReset
};

export const ModelsReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.POPULATE_MODELLIST:
      return {
        ...state,
        ModelList: action.models.length === 0
          ? []
          : action.models
      }
    default:
      return state
  }
}

export default ModelsReducer