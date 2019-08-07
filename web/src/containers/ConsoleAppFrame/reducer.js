import { Types } from "./actions"


const initialState = {
  mobileOpen: false,
  headerTitle: "Overview"
};

export const ConsoleAppFrame = (state = initialState, action) => {
  switch (action.type) {
    case Types.DRAWER_TOGGLE :
      return {
        ...state,
        mobileOpen: !state.mobileOpen
      }
    case Types.CHANGE_HEADER_TITLE :
      return {
        ...state,
        headerTitle: action.headerTitle
      }
    default:
      return state
  }
}
