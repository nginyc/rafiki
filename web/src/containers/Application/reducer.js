import { Types } from "./actions"

const StatesToReset = {
}

const initialState = {
  // MODEL-List
  ApplicationList: [],
  SelectedApplication: {},
  ...StatesToReset
};

export const ApplicationsReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.POPULATE_INFERENCEJOB:
      return {
        ...state,
        ApplicationList: action.jobs.length === 0
          ? []
          : action.jobs
      }
    default:
      return state
  }
}

export default ApplicationsReducer