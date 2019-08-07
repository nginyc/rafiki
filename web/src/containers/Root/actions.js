export const Types = {
  NOTIFICATION_SHOW: "NOTIFICATION_SHOW",
  NOTIFICATION_HIDE: "NOTIFICATION_HIDE",

  AUTH_START: "AUTH_START",
  AUTH_SUCCESS: "AUTH_SUCCESS",
  AUTH_FAIL: "AUTH_FAIL",
  AUTH_LOGOUT: "AUTH_LOGOUT",
  // for sagas
  SIGN_IN_REQUEST: "SIGN_IN_REQUEST",
  AUTH_CHECK_STATE: "AUTH_CHECK_STATE",
  // for appbar menuitem
  LOGIN_MENU_OPEN: "LOGIN_MENU_OPEN",
  LOGIN_MENU_CLOSE: "LOGIN_MENU_CLOSE",
  // landing drawer toggle
  DRAWER_TOGGLE: "root/drawer_toggle",
}

export const handleDrawerToggle = () => ({
  type: Types.DRAWER_TOGGLE
});

export const loginMenuOpen = anchorElId => ({
  type: Types.LOGIN_MENU_OPEN,
  anchorElId
});

export const loginMenuClose = () => ({
  type: Types.LOGIN_MENU_CLOSE
});

// for notification area
export const notificationShow = message => ({
  type: Types.NOTIFICATION_SHOW,
  message
});

export const notificationHide = () => ({
  type: Types.NOTIFICATION_HIDE
});

// for authentication actions
export const authStart = () => ({
  type: Types.AUTH_START
})

export const authSuccess = (token, user_id) => ({
  type: Types.AUTH_SUCCESS,
  token,
  user_id
})

export const authFail = error => ({
  type: Types.AUTH_FAIL,
  error
})

export const logout = () => ({
    type: Types.AUTH_LOGOUT
})

// for sagas
export const signInRequest = (authData) => ({
  type: Types.SIGN_IN_REQUEST,
  authData
});

export const authCheckState = () => ({
  type: Types.AUTH_CHECK_STATE
})
