import { watchSignInRequest, authLogin } from '../AppRoot'
import * as actions from '../../containers/Root/actions'

import { cloneableGenerator } from '@redux-saga/testing-utils'
import { takeLatest, put, call, all } from 'redux-saga/effects';
import * as api from "../../services/AuthAPI"

describe("watchSigninRequest should wait for action SIGN_IN_REQUEST", function() {
    const gen = watchSignInRequest()
    const actual = gen.next()
    expect(actual.value).toEqual(takeLatest(actions.Types.SIGN_IN_REQUEST, authLogin))
})

describe("authLogin Saga Unit Test", function() {
    const gen = cloneableGenerator(authLogin)(actions.signInRequest({ username: "superadmin@rafiki ", password: "rafiki" }))

    it("should dispatch action AUTH_START", function() { 
        const step1 = gen.next() 
        expect(step1.value).toEqual(put(actions.authStart()))
    });

    it("should be able to call request the token", function() {
        const step2 = gen.next()
        expect(step2.value).toEqual(call(api.requestSignIn, {
            username: "superadmin@rafiki ",
            password: "rafiki"
        }))
    });

    it("should be able to dispatch AUTH_SUCCESS action on success", function(){
        const step3 = gen.next({data: { token: "foobar"}})
        expect(step3.value).toEqual(all([put(actions.notificationShow("Successfully signed in")), put(actions.authSuccess("foobar"))]))
    })

    it("should be able to catch error on response and dispatch show error's actions", function(){
        const error = new Error("custom error")
        const stepError = gen.throw(error)
        expect(stepError.value).toEqual(put(actions.authFail(error)))
        const stepError2 = gen.next()
        expect(stepError2.value).toEqual(put(actions.notificationShow("Failed to sign in")))
    });

    it("should be able to call action redirect to dataset page", function() {
        //TODO
    })
}) 