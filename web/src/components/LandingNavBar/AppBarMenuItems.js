import PropTypes from "prop-types";
import React, { Fragment } from "react";
import { Link } from "react-router-dom";
// and AppBar right-most icon
// import MenuList from '@material-ui/core/MenuList';
import MenuItem from "@material-ui/core/MenuItem";


const linkStyle = {
  textDecoration: "none"
};

const AppBarMenuItems = ({ onClick, logout, isAuth }) => (
  <Fragment>
    <Link style={linkStyle} to={`/#/profile/${isAuth}`}>
      <MenuItem
        onClick={() => {
          onClick();
        }}
      >
        My account
      </MenuItem>
    </Link>
    <MenuItem
      onClick={() => {
        onClick();
        logout();
      }}
    >
      Logout
    </MenuItem>
  </Fragment>
);

AppBarMenuItems.propTypes = {
  onClick: PropTypes.func,
  logout: PropTypes.func,
  isAuth: PropTypes.any
};

export default AppBarMenuItems;
