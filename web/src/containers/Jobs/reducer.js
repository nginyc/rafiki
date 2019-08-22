import * as actions from "./actions"

const initialState = {
    jobsList: []
}

export function JobsReducer(state = initialState, action) {
    switch (action.type) {
        case actions.Types.POPULATE_TRAINJOBSLIST:
            return {
                ...state,
                jobsList: action.jobsList
            }
        case actions.Types.POPULATE_TRIALSTOJOBS:
            const new_state = {
                ...state,
                jobsList: state.jobsList.map((job) => {
                    // eslint-disable-next-line
                    if (job.app === action.app && job.app_version == action.appVersion) { // JavaScript has both strict and type–converting comparisons. A strict comparison (e.g., ===) is only true if the operands are of the same type and the contents match. The more commonly-used abstract comparison (e.g. ==) converts the operands to the same type before making the comparison. For relational abstract comparisons (e.g., <=), the operands are first converted to primitives, then to the same type, before comparison.
                        return {
                            ...job,
                            trials: action.trials
                        }
                    } else {
                        return job
                    }
                })
            }
            console.log("POPULATE_TRIALSTOJOBS", new_state)
            return new_state
        default:
            return state
    }
}

export default JobsReducer