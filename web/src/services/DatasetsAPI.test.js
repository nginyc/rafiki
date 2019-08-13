import { _makeUrl, requestDatasetList } from "./ClientAPI"
import HTTPconfig from "HTTPconfig"

describe("request ListDataset should return an axios promise", function() {
    // TODO, Think about how to test 

})

describe("post CreatDataset should return an axios promise", function() {
    // TODO
})

describe("_makeUrl should configure the urlPath correctly", function() {
    const urlPath = "datasets"
    it("should be able to return url without params", function() {
        const actual = _makeUrl(urlPath)
        expect(actual).toEqual(`${HTTPconfig.gateway}datasets`)
    })

    it("should be able to return url with prarams if params is given", function() {
        const params = {task: "IMAGE_CLASSIFICATION"}
        const actual = _makeUrl(urlPath, params)
        expect(actual).toEqual(`${HTTPconfig.gateway}datasets?task=IMAGE_CLASSIFICATION`)
    })

})