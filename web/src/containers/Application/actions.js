export const Types = {
    FETCH_GET_INFERENCEJOB: "Application/fetch_get_inferencejob",
    POST_CREATE_INFERENCEJOB: "Application/post_create_inferencejob",

    POPULATE_INFERENCEJOB: "Application/display_infenrencejob",
}

export const fetchGetInferencejob = () => {
    return {
        types: Types.FETCH_GET_INFERENCEJOB
    }
}

export const postCreateInferenceJob = () => {
    return {
        types: Types.POST_CREATE_INFERENCEJOB
    }
}

// sync
export const populateInferenceJob = (jobs) => {
    return {
        types: Types.POPULATE_INFERENCEJOB,
            jobs
    }
}