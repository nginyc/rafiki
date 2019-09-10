import { DatasetsReducer } from './reducer'
import { populateDSList, Types } from './actions'

describe("Unit Test for Datasets Reducer", function() {
    const initialState = {
        DatasetList: [] 
    }

    const DatasetList = [
        {
            "datetime_created": "Wed, 19 Jun 2019 13:31:56 GMT",
            "id": "59fe7d7a-7056-4f40-bafc-35da06e08766",
            "name": "fashion_minist_app_train",
            "size_bytes": 34864119,
            "task": "IMAGE_CLASSIFICATION"
        },
        {
            "datetime_created": "Wed, 19 Jun 2019 13:32:23 GMT",
            "id": "65500bed-338e-4491-9761-dbbecb811c90",
            "name": "fashion_minist_app_test",
            "size_bytes": 6116386,
            "task": "IMAGE_CLASSIFICATION"
        }]

    const action = populateDSList(DatasetList)

    it("should be able return new dataList if action POPULATE_DS_LIST is dispatched", function() {
        const actual = DatasetsReducer(initialState, action)
        expect(actual).toEqual({
            DatasetList: [{
                "datetime_created": "Wed, 19 Jun 2019 13:31:56 GMT",
                "id": "59fe7d7a-7056-4f40-bafc-35da06e08766",
                "name": "fashion_minist_app_train",
                "size_bytes": 34864119,
                "task": "IMAGE_CLASSIFICATION"
            },{
                "datetime_created": "Wed, 19 Jun 2019 13:32:23 GMT",
                "id": "65500bed-338e-4491-9761-dbbecb811c90",
                "name": "fashion_minist_app_test",
                "size_bytes": 6116386,
                "task": "IMAGE_CLASSIFICATION"
            }]
        })
    })
})