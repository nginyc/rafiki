import React, {Fragment} from 'react';
import { withRouter } from 'react-router-dom'
import { connect } from "react-redux"
import { compose } from "redux"

import AppBar from '@material-ui/core/AppBar';
import Grid from '@material-ui/core/Grid';
import Hidden from '@material-ui/core/Hidden';
import MenuIcon from '@material-ui/icons/Menu';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';

// for login menu
import IconButton from "@material-ui/core/IconButton";
import Menu from "@material-ui/core/Menu";
import AppBarMenuItems from "../../LandingNavBar/AppBarMenuItems"

// Icons
import ExitToApp from "@material-ui/icons/ExitToApp";

import * as actions from "../../../containers/Root/actions"


const lightColor = 'rgba(255, 255, 255, 0.7)';

const styles = theme => ({
  appBar: {
    backgroundColor: "#eaeff1",
    color: "#000"
  },
  avatar: {
    margin: 10,
    color: '#fff',
    backgroundColor: theme.palette.secondary.main,
  },
  menuButton: {
    marginLeft: -theme.spacing(1),
  },
  iconButtonAvatar: {
    padding: 4,
  },
  link: {
    textDecoration: 'none',
    color: lightColor,
    '&:hover': {
      color: theme.palette.common.white,
    },
  },
  button: {
    borderColor: lightColor,
  },
});

class Header extends React.Component {
  handleMenuOpen = event => {
    this.props.loginMenuOpen(event.currentTarget.id);
  };

  handleMenuClose = () => {
    this.props.loginMenuClose();
  };

  handleLogout = () => {
    console.log("logging out, clearing token")
    localStorage.removeItem('token');
    localStorage.removeItem('expirationDate');
    this.props.history.push(`/`);
    window.location.reload();
  }

  render() {
    const {
      classes,
      title,
      onDrawerToggle,
      isAuthenticated,
      anchorElId,
      //initials,
      //bgColor
    } = this.props;
  
    return (
      <Fragment>
        <AppBar className={classes.appBar} position="sticky" elevation={0}>
          <Toolbar>
            <Grid container spacing={8} alignItems="center">
              <Hidden mdUp>
                <Grid item>
                  <IconButton
                    color="inherit"
                    aria-label="Open drawer"
                    onClick={onDrawerToggle}
                    className={classes.menuButton}
                  >
                    <MenuIcon />
                  </IconButton>
                </Grid>
              </Hidden>
              <Grid item sm>
                <Typography color="inherit" variant="h5">
                  {title}
                </Typography>
              </Grid>
              <Grid item>
                <Fragment>
                  <IconButton
                    aria-haspopup="true"
                    aria-label="More"
                    aria-owns="Open right Menu"
                    color="inherit"
                    id="loginMenuButton"
                    onClick={this.handleMenuOpen}
                    className={classes.iconButtonAvatar}
                  >
                    <ExitToApp />
                  </IconButton>
                  <Menu
                    anchorEl={
                      (anchorElId && document.getElementById(anchorElId)) ||
                      document.body
                    }
                    id="menuRight"
                    onClose={this.handleMenuClose}
                    open={!!anchorElId}
                  >
                    <AppBarMenuItems
                      isAuth={isAuthenticated}
                      logout={this.handleLogout}
                      onClick={this.handleMenuClose}
                    />
                  </Menu>
                </Fragment>
              </Grid>
            </Grid>
          </Toolbar>
        </AppBar>
      </Fragment>
    )
  }
}

const mapStateToProps = state => ({
  anchorElId: state.Root.dropdownAnchorElId,
  isAuthenticated: state.Root.token !== null,
  // initials: state.firebaseReducer.profile.initials,
  // bgColor: state.firebaseReducer.profile.color
})

const mapDispatchToProps = {
  loginMenuOpen: actions.loginMenuOpen,
  loginMenuClose: actions.loginMenuClose,
}

export default compose(
  connect(
    mapStateToProps,
    mapDispatchToProps
  ),
  withRouter,
  withStyles(styles)
)(Header)
