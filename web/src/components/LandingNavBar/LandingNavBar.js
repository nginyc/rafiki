import React, { Fragment } from "react";
import { Link, withRouter } from 'react-router-dom'
import { compose } from "redux";
import { connect } from "react-redux";

import { withStyles } from '@material-ui/core/styles';
import Typography from "@material-ui/core/Typography"
import AppBar from '../LandingComponents/AppBar';
import Grid from '@material-ui/core/Grid';
import Hidden from '@material-ui/core/Hidden';
import MenuIcon from '@material-ui/icons/Menu';
import LandingNavigator from "./LandingNavigator"

// for login menu
import Button from "@material-ui/core/Button";
import IconButton from "@material-ui/core/IconButton";
import Menu from "@material-ui/core/Menu";
import Avatar from '@material-ui/core/Avatar';
import AppBarMenuItems from "./AppBarMenuItems"

import Toolbar, { styles as toolbarStyles } from '../LandingComponents/Toolbar';
import Logo from "../../assets/LOGO_Rafiki-4.svg"

import * as actions from "../../containers/Root/actions"

const styles = theme => ({
  LandingAppBar: {
    // borderBottom: `1px solid ${theme.palette.border}`,
    // backgroundColor: theme.palette.common.white,
    zIndex: theme.zIndex.drawer + 1,
  },
  title: {
    font: '500 25px Roboto,sans-serif',
    cursor: "pointer",
    color: "#FFF",
    textDecoration: "none",
    marginRight: 20
  },
  placeholder: toolbarStyles(theme).root,
  toolbar: {
    justifyContent: 'space-between',
  },
  left: {
    flex: 1,
    display: 'flex',
    justifyContent: 'flex-start',
    alignItems: "center"
  },
  logo: {
    height: 36,
    marginRight: 10
  },
  leftLinkActive: {
    color: theme.palette.secondary.dark,
  },
  right: {
    flex: 1,
    display: 'flex',
    justifyContent: 'flex-end',
  },
  rightLink: {
    font: '300 18px Roboto,sans-serif',
    color: theme.palette.common.white,
    marginLeft: theme.spacing(1) * 5,
    textDecoration: "none",
    '&:hover': {
      color: theme.palette.secondary.main,
    },
  },
  rightLinkActive: {
    font: '300 18px Roboto,sans-serif',
    color: theme.palette.secondary.main,
    marginLeft: theme.spacing(1) * 5,
    textDecoration: "none",
  },
  linkSecondary: {
    color: theme.palette.secondary.main,
  },
  avatar: {
    margin: 10,
    color: '#fff',
    backgroundColor: theme.palette.secondary.main,
  },
  iconButtonAvatar: {
    padding: 4,
    marginLeft: theme.spacing(1) * 3,
    textDecoration: "none"
  },
  menuButton: {
    marginLeft: -theme.spacing(1),
    marginRight: theme.spacing(1) * 2,
  },
});

class LandingNavBar extends React.Component {
  handleMenuOpen = event => {
    this.props.loginMenuOpen(event.currentTarget.id);
  };

  handleMenuClose = () => {
    this.props.loginMenuClose();
  };

  handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('expirationDate');
    this.props.history.push(`/`);
    window.location.reload();
  };

  render() {
    const {
      anchorElId,
      isAuthenticated,
      classes,
      handleDrawerToggle,
      RootMobileOpen,
      location
    } = this.props;

    const links = isAuthenticated
      ? (
        <Fragment>
          <Typography
            variant="h6"
          >
            <Link to="/console/datasets/list-dataset" className={classes.rightLink}>
              {'Go To Console'}
            </Link>
          </Typography>
          <IconButton
            aria-haspopup="true"
            aria-label="More"
            aria-owns="Open right Menu"
            color="inherit"
            id="loginMenuButton"
            onClick={this.handleMenuOpen}
            className={classes.iconButtonAvatar}
          >
            <Avatar
              className={classes.avatar}
              style={{
                backgroundColor: "orange" //bgColor
              }}
            >
              {"LQ" /*initials*/}
            </Avatar>
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
      )
      : (
        <Fragment>
          <Button
            color="inherit"
            style={{
              textDecoration: "none",
              fontSize: 16,
            }}
            component={Link}
            to={"/sign-in"}
          >
            {"Sign in"}
          </Button>
        </Fragment>
      )

    const navLinks = [
      /*{
        url: "/publications",
        label: "Publicationcs",
      },*/
      {
        url: "/contact",
        label: "Contact",
      },
      {
        url: "https://nginyc.github.io/rafiki/docs/latest/src/user/index.html",
        label: "Docs",
      },
    ]

    return (
      <div>
        <LandingNavigator
          PaperProps={{ style: { width: 250, backgroundColor: "rgb(0,0,0)" } }}
          variant="temporary"
          open={RootMobileOpen}
          onClose={handleDrawerToggle}
        />
        <AppBar position="fixed" className={classes.LandingAppBar}>
          <Toolbar className={classes.toolbar}>
            <Hidden mdUp>
              <Grid item>
                <IconButton
                  color="inherit"
                  aria-label="Open drawer"
                  onClick={handleDrawerToggle}
                  className={classes.menuButton}
                >
                  <MenuIcon />
                </IconButton>
              </Grid>
            </Hidden>
            <div className={classes.left}>
              <Link to="/">
                <img alt="logo" src={Logo} className={classes.logo} />
              </Link>
              <Link to="/" className={classes.title}>
                {'Rafiki'}
              </Link>
              <Hidden smDown>
                {navLinks.map((link, index) => (
                  /^https?:\/\//.test(link.url) ? // test if the url is external 
                    <a key={index} href={link.url} className={
                      location.pathname === link.url
                        ? (
                          classes.rightLinkActive
                        )
                        : (
                          classes.rightLink
                        )
                    }>
                      {link.label}
                    </a>
                    :
                    <Link
                      key={index}
                      to={link.url}
                      className={
                        location.pathname === link.url
                          ? (
                            classes.rightLinkActive
                          )
                          : (
                            classes.rightLink
                          )
                      }
                    >
                      {link.label}
                    </Link>
                ))}
              </Hidden>
            </div>
            {links}
          </Toolbar>
        </AppBar>
        <div className={classes.placeholder} />
      </div>
    )
  }
}


const mapStateToProps = state => ({
  anchorElId: state.Root.dropdownAnchorElId,
  isAuthenticated: state.Root.token !== null,
  // initials: state.firebaseReducer.profile.initials,
  // bgColor: state.firebaseReducer.profile.color
  RootMobileOpen: state.Root.RootMobileOpen,
});

const mapDispatchToProps = {
  loginMenuOpen: actions.loginMenuOpen,
  loginMenuClose: actions.loginMenuClose,
  handleDrawerToggle: actions.handleDrawerToggle
}


export default compose(
  connect(
    mapStateToProps,
    mapDispatchToProps
  ),
  withRouter,
  withStyles(styles)
)(LandingNavBar);
