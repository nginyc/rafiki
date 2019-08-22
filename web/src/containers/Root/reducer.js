import { Types } from "./actions"


const initialState = {
    token: null,
    user_id: null, 
    error: null, 
    loading: false,
    notification: {
      show: false,
      message: ""
    },
    dropdownAnchorElId: false,
    RootMobileOpen: false,
}


export const Root = (state = initialState, action) => {
  switch (action.type) {
    // login menu on appbar
    case Types.LOGIN_MENU_OPEN:
      return {
        ...state,
        dropdownAnchorElId: action.anchorElId
      };
    case Types.LOGIN_MENU_CLOSE:
      return {
        ...state,
        dropdownAnchorElId: false
      };
    // for authentications
    case Types.AUTH_START:
      return {
        ...state,
        error: null,
        loading: true
      }
    case Types.AUTH_SUCCESS:
      return {
        ...state,
        token: action.token,
        user_id: action.user_id,
        error: null,
        loading: false
      }
    case Types.AUTH_FAIL:
      return {
        ...state,
        error: action.error,
        loading: false
      }
    case Types.AUTH_LOGOUT:
      return {
        ...state,
        token: null
      }
    // for notification area
    case Types.NOTIFICATION_SHOW:
      return {
        ...state,
        notification: {
          show: true,
          message: action.message
        }
      };
    case Types.NOTIFICATION_HIDE:
      return {
        ...state,
        notification: {
          show: false,
          message: ""
        }
      };
    // for landing page navbar drawer
    case Types.DRAWER_TOGGLE :
      return {
        ...state,
        RootMobileOpen: !state.RootMobileOpen
      };
    default:
      return state;
  }
};
