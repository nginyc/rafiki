export const Types = {
   // ==== MODELS ===== 
   REQUEST_MODEL_LIST:"Jobs/request_train_jobslist", // async action is often named with three _ in between
}

export function requestModelList() {
    return {
        type: Types.REQUEST_MODEL_LIST
    }
}

