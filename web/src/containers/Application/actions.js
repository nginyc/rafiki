export const Types = {
    FETCH_GET_INFERENCEJOB: "Application/fetch_get_inferencejob",
    POST_CREATE_INFERENCEJOB: "Application/post_create_inferencejob",

    GET_RUNNING_INFERENCEJOB: "Application/get_running_inferencejob",
    SELECT_INFERENCEJOB: "Application/select_inferencejob",

    POPULATE_INFERENCEJOB: "Application/display_infenrencejob",
}

export const fetchGetInferencejob = () => {
    return {
        type: Types.FETCH_GET_INFERENCEJOB
    }
}

export const postCreateInferenceJob = (app, appVersion, budget) => {
    return {
        type: Types.POST_CREATE_INFERENCEJOB,
        app,
        appVersion,
        budget
    }
}

// sync
export const populateInferenceJob = (jobs) => {
    return {
        type: Types.POPULATE_INFERENCEJOB,
        jobs
    }
}