export const Types = {
  DRAWER_TOGGLE: "ConsoleAppFrame/drawer_toggle",
  CHANGE_HEADER_TITLE: "ConsoleAppFrame/change_header_title",
  // reset loading-bar status when route change
  RESET_LOADING_BAR: "ConsoleAppFrame/reset_loading_bar",
}

export const handleDrawerToggle = () => ({
  type: Types.DRAWER_TOGGLE
})

export const handleHeaderTitleChange = headerTitle => ({
  type: Types.CHANGE_HEADER_TITLE,
  headerTitle
})

// clear redux-loading-bar when route unmount
export const resetLoadingBar = () => ({
  type: Types.RESET_LOADING_BAR
})
