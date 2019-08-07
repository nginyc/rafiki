import jobsReducer from "./reducer"
import * as actions from "./actions"

describe("Unit Test reducer", function() {
    const initialState = { jobsList: [] }

    const jobsList = [
        {
            "app": "food101",
            "app_version": 1,
            "budget": {
                "GPU_COUNT": 1,
                "TIME_HOURS": 12
            },
            "datetime_started": "Tue, 30 Jul 2019 14:24:37 GMT",
            "datetime_stopped": "Tue, 30 Jul 2019 22:01:57 GMT",
            "id": "3ac7c4b0-bd25-4989-9c7a-d7fa4af28f2c",
            "status": "ERRORED",
            "task": "IMAGE_CLASSIFICATION",
            "train_args": {},
            "train_dataset_id": "0278939f-77ee-4311-bfb7-d0660d751670",
            "val_dataset_id": "0eef229d-38cf-4970-a4f7-d1ad60bd0ffe"
        }]

    const action = actions.populate_trainjobslist(jobsList)

    it("should be able to return new jobslist when POPULATE_TRAINJOBSLIST is dispatched", function() {
        const actual = jobsReducer(initialState, action) 
        expect(actual).toEqual({
            jobsList:  [
                {
                    "app": "food101",
                    "app_version": 1,
                    "budget": {
                        "GPU_COUNT": 1,
                        "TIME_HOURS": 12
                    },
                    "datetime_started": "Tue, 30 Jul 2019 14:24:37 GMT",
                    "datetime_stopped": "Tue, 30 Jul 2019 22:01:57 GMT",
                    "id": "3ac7c4b0-bd25-4989-9c7a-d7fa4af28f2c",
                    "status": "ERRORED",
                    "task": "IMAGE_CLASSIFICATION",
                    "train_args": {},
                    "train_dataset_id": "0278939f-77ee-4311-bfb7-d0660d751670",
                    "val_dataset_id": "0eef229d-38cf-4970-a4f7-d1ad60bd0ffe"
                }]
        }) 
     })
})
