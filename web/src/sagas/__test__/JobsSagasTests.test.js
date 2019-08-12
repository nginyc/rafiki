import * as actions from 'containers/Jobs/actions'
import { takeLatest, call, put } from "redux-saga/effects";
import { getJobsList, watchRequestJobsList, postTrainJob, watchPostTrainJob, watchRequestTrialsListOfJob, getTrialsListOfJob} from '../JobsSagas'
import { notificationShow } from 'containers/Root/actions';
import * as api from 'services/ClientAPI';

describe("Jobs Saga", function () {
    describe("watch requestJobDetails", function () {
        it("should be able to watch action requestJobDetails", function () {
            const gen = watchRequestJobsList()
            const actual = gen.next()
            expect(actual.value).toEqual(takeLatest(actions.Types.REQUEST_TRAIN_JOBSLIST, getJobsList))
        })
    })

    describe("getJobsDetails", function () {
        const gen = getJobsList()

        it("should be able to get jobs details (call dataset API)", function () {
            const step1 = gen.next()
            const step2 = gen.next()
            const step3 = gen.next("some token")
            const actual = gen.next("some id") 
            expect(actual.value).toEqual(call(api.requestTrainJobsList, {user_id: "some id"}, "some token"))
        })

        it("should be able to catch error on response and dispatch show error's actions", function() {
            const error = new Error("custom error")
            const stepError = gen.throw(error)
            expect(stepError.value).toEqual(put(notificationShow("Failed to Fetch TrainJobsList")))
        })
    })

    describe("watchRequestTrialsListOfJob", function() {
        it("should be able to call requestTrialsList API", function() {
            const gen = watchRequestTrialsListOfJob()
            const actual = gen.next()
            expect(actual.value).toEqual(takeLatest(actions.Types.REQUEST_TRIALSLIST_OFJOB, getTrialsListOfJob))
        })
    })

    describe("watch create new train jobs", function() {
        it("should be able to watch action postTrainJob", function() {
            const gen = watchPostTrainJob()
            const actual = gen.next()
            expect(actual.value).toEqual(takeLatest(actions.Types.POST_CREAT_TRAINJOB, postTrainJob))
        })
    })

    describe("postCreateTrainJob", function() {
        it("should be able to post Trainjob to the server", function() {
            // TODO
        })
    })
})