import jobsReducer from "./reducer"
import * as actions from "./actions"

describe("Unit Test reducer", function () {

    describe("polulate_trainjobslist", function () {
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

        const action = actions.populateJobsList(jobsList)

        it("should be able to return new jobslist when POPULATE_TRAINJOBSLIST is dispatched", function () {
            const actual = jobsReducer(initialState, action)
            expect(actual).toEqual({
                "jobsList": [
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

    describe("populate_trialtojobs", function () {

        const initialState = {
            "jobsList": [
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
        } 

        const trials = [{"datetime_started":"Tue, 30 Jul 2019 22:00:53 GMT","datetime_stopped":"Tue, 30 Jul 2019 22:01:57 GMT","id":"58a362a3-5e5e-4d73-b858-20c4650e8e84","is_params_saved":false,"model_name":"TfXception","no":8,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.00122070731480779,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"58a362a3-5e5e-4d73-b858-20c4650e8e84","trial_no":8},"score":null,"status":"ERRORED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 20:54:59 GMT","datetime_stopped":"Tue, 30 Jul 2019 22:00:53 GMT","id":"fece1c68-1129-410a-a979-f28831778709","is_params_saved":true,"model_name":"TfXception","no":7,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.002108470381102979,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"fece1c68-1129-410a-a979-f28831778709","trial_no":7},"score":0.6575,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 19:49:41 GMT","datetime_stopped":"Tue, 30 Jul 2019 20:54:59 GMT","id":"8da2aeaa-b7bc-4308-b1b9-58a96552a8d6","is_params_saved":true,"model_name":"TfXception","no":6,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0028047149427815987,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"8da2aeaa-b7bc-4308-b1b9-58a96552a8d6","trial_no":6},"score":0.658333333333333,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 18:44:10 GMT","datetime_stopped":"Tue, 30 Jul 2019 19:49:41 GMT","id":"3ef10686-e130-4246-b34a-b32374501af1","is_params_saved":true,"model_name":"TfXception","no":5,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0026132985078009805,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"3ef10686-e130-4246-b34a-b32374501af1","trial_no":5},"score":0.655833333333333,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 17:39:08 GMT","datetime_stopped":"Tue, 30 Jul 2019 18:44:10 GMT","id":"50bb0201-7672-485a-985a-6054a86aeda3","is_params_saved":true,"model_name":"TfXception","no":4,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0023964292882948084,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"50bb0201-7672-485a-985a-6054a86aeda3","trial_no":4},"score":0.555833333333333,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 16:33:45 GMT","datetime_stopped":"Tue, 30 Jul 2019 17:39:08 GMT","id":"50b3ee12-5a8b-4b25-a9e0-10fad6a84dd8","is_params_saved":true,"model_name":"TfXception","no":3,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.001767142405280325,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"50b3ee12-5a8b-4b25-a9e0-10fad6a84dd8","trial_no":3},"score":0.569166666666667,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 15:28:53 GMT","datetime_stopped":"Tue, 30 Jul 2019 16:33:45 GMT","id":"f00cb9bc-926b-4702-a1a7-0d7592623934","is_params_saved":true,"model_name":"TfXception","no":2,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0015810119724999159,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"f00cb9bc-926b-4702-a1a7-0d7592623934","trial_no":2},"score":0.686666666666667,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 14:25:01 GMT","datetime_stopped":"Tue, 30 Jul 2019 15:28:53 GMT","id":"79bd6c8b-0dd2-4a81-8ba6-e2850efc3a3a","is_params_saved":true,"model_name":"TfXception","no":1,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0011452099021191419,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"79bd6c8b-0dd2-4a81-8ba6-e2850efc3a3a","trial_no":1},"score":0.595,"status":"COMPLETED","worker_id":"557316a955b5"}]
 
        const app = "food101"
        const appVersion = 1
        const action = actions.populateTrialsToJobs(trials, app, appVersion)

        it("shoulde ba able to fetch trial details", function () {

            const actual = jobsReducer(initialState, action)
            expect(actual).toEqual({
                jobsList: [{
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
                    "val_dataset_id": "0eef229d-38cf-4970-a4f7-d1ad60bd0ffe",
                    "trials": [{"datetime_started":"Tue, 30 Jul 2019 22:00:53 GMT","datetime_stopped":"Tue, 30 Jul 2019 22:01:57 GMT","id":"58a362a3-5e5e-4d73-b858-20c4650e8e84","is_params_saved":false,"model_name":"TfXception","no":8,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.00122070731480779,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"58a362a3-5e5e-4d73-b858-20c4650e8e84","trial_no":8},"score":null,"status":"ERRORED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 20:54:59 GMT","datetime_stopped":"Tue, 30 Jul 2019 22:00:53 GMT","id":"fece1c68-1129-410a-a979-f28831778709","is_params_saved":true,"model_name":"TfXception","no":7,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.002108470381102979,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"fece1c68-1129-410a-a979-f28831778709","trial_no":7},"score":0.6575,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 19:49:41 GMT","datetime_stopped":"Tue, 30 Jul 2019 20:54:59 GMT","id":"8da2aeaa-b7bc-4308-b1b9-58a96552a8d6","is_params_saved":true,"model_name":"TfXception","no":6,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0028047149427815987,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"8da2aeaa-b7bc-4308-b1b9-58a96552a8d6","trial_no":6},"score":0.658333333333333,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 18:44:10 GMT","datetime_stopped":"Tue, 30 Jul 2019 19:49:41 GMT","id":"3ef10686-e130-4246-b34a-b32374501af1","is_params_saved":true,"model_name":"TfXception","no":5,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0026132985078009805,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"3ef10686-e130-4246-b34a-b32374501af1","trial_no":5},"score":0.655833333333333,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 17:39:08 GMT","datetime_stopped":"Tue, 30 Jul 2019 18:44:10 GMT","id":"50bb0201-7672-485a-985a-6054a86aeda3","is_params_saved":true,"model_name":"TfXception","no":4,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0023964292882948084,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"50bb0201-7672-485a-985a-6054a86aeda3","trial_no":4},"score":0.555833333333333,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 16:33:45 GMT","datetime_stopped":"Tue, 30 Jul 2019 17:39:08 GMT","id":"50b3ee12-5a8b-4b25-a9e0-10fad6a84dd8","is_params_saved":true,"model_name":"TfXception","no":3,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.001767142405280325,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"50b3ee12-5a8b-4b25-a9e0-10fad6a84dd8","trial_no":3},"score":0.569166666666667,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 15:28:53 GMT","datetime_stopped":"Tue, 30 Jul 2019 16:33:45 GMT","id":"f00cb9bc-926b-4702-a1a7-0d7592623934","is_params_saved":true,"model_name":"TfXception","no":2,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0015810119724999159,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"f00cb9bc-926b-4702-a1a7-0d7592623934","trial_no":2},"score":0.686666666666667,"status":"COMPLETED","worker_id":"557316a955b5"},{"datetime_started":"Tue, 30 Jul 2019 14:25:01 GMT","datetime_stopped":"Tue, 30 Jul 2019 15:28:53 GMT","id":"79bd6c8b-0dd2-4a81-8ba6-e2850efc3a3a","is_params_saved":true,"model_name":"TfXception","no":1,"proposal":{"knobs":{"batch_size":8,"learning_rate":0.0011452099021191419,"max_epochs":30,"max_image_size":299},"meta":{"proposal_type":"SEARCH"},"params_type":"NONE","to_cache_params":false,"to_eval":true,"to_save_params":true,"trial_id":"79bd6c8b-0dd2-4a81-8ba6-e2850efc3a3a","trial_no":1},"score":0.595,"status":"COMPLETED","worker_id":"557316a955b5"}]
        }]
    })

    })
})
})
