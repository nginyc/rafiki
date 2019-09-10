export const Types = {
    // ==== MODELS ===== 
    // ASYNC
    REQUEST_MODEL_LIST: "Models/request_model_list",
    POST_CREATE_MODEL: "Models/post_create_model",
    // SYNC
    POPULATE_MODELLIST: "Models/populate_modellist",
}

export function requestModelList() {
    return {
        type: Types.REQUEST_MODEL_LIST
    }
}

export function populateModelList(models) {
    return {
        type: Types.POPULATE_MODELLIST,
        models
    }
}
