import { Root } from "./reducer"
import * as actions from "./actions"

describe("Root Reducer Unit Tests", function () {
    it("should return the initial state", function () {
        expect(Root(undefined, {})).toEqual(
            {
                token: null,
                error: null,
                loading: false,
                notification: {
                    show: false,
                    message: ""
                },
                dropdownAnchorElId: false,
                RootMobileOpen: false,
            }
        )
    })

    it("should handle AUTH_START", function () {
        const actual = Root(undefined, {
            type: actions.Types.AUTH_START
        })
        expect(actual).toEqual(
            {
                token: null,
                error: null,
                loading: true,
                notification: {
                    show: false,
                    message: ""
                },
                dropdownAnchorElId: false,
                RootMobileOpen: false,
            })
    })
    it("should handle AUTH_SUCCESS", () => {
        const actual = Root(undefined, {
            type: actions.Types.AUTH_SUCCESS,
            token: "Mock Token"
        })
        expect(actual).toEqual( {
            token: "Mock Token",
            error: null,
            loading: false,
            notification: {
                show: false,
                message: ""
            },
            dropdownAnchorElId: false,
            RootMobileOpen: false,
        })
     })

    it("should handle AUTH_FAIL", () => {
        const actual = Root(undefined, {
            type: actions.Types.AUTH_FAIL,
            error: "Mock Error"
        })
        
        expect(actual).toEqual( {
            token: null,
            error: "Mock Error",
            loading: false,
            notification: {
                show: false,
                message: ""
            },
            dropdownAnchorElId: false,
            RootMobileOpen: false,
        })
    })
    it("should handle AUTH_LOGOUT", () => { 
        const actual = Root({
            token: "Mock Token",
            error: null,
            loading: false,
            notification: {
                show: false,
                message: ""
            },
            dropdownAnchorElId: false,
            RootMobileOpen: false,
        },{
            type:actions.Types.AUTH_LOGOUT,
        })
        expect(actual).toEqual(
            {
                token: null,
                error: null,
                loading: false,
                notification: {
                    show: false,
                    message: ""
                },
                dropdownAnchorElId: false,
                RootMobileOpen: false,
            }
        )
    })
})