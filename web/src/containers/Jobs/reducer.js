import * as actions from "./actions"

const initialState = {
    jobsList:[]
}

export function JobsReducer(state = initialState, action) {
    switch (action.type) {
        case actions.Types.POPULATE_TRAINJOBSLIST:
            return {
                ...state,
                jobsList: action.jobsList
            } 
        default: 
            return state
    }
}

export default JobsReducer