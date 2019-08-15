import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { compose } from "redux";
import { Link, withRouter } from "react-router-dom";

import { withStyles } from '@material-ui/core/styles';

import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

import Logo from "../../assets/Logo-Rafiki-cleaned.png"


// Navigator basic color dark blue specified in
// ConsoleTheme MuiDrawer's paper
const styles = theme => ({
  categoryHeader: {
    paddingTop: 16,
    paddingBottom: 16,
  },
  categoryHeaderPrimary: {
    color: theme.palette.common.white,
  },
  categoryHeaderPrimaryActive: {
    color: 'inherit'
  },
  item: {
    paddingTop: 11,
    paddingBottom: 11,
    color: 'rgba(255, 255, 255, 0.7)',
  },
  itemCategory: {
    backgroundColor: '#232f3e',
    boxShadow: '0 -1px 0 #404854 inset',
    paddingTop: 16,
    paddingBottom: 16,
  },
  firebase: {
    fontSize: 24,
    fontFamily: theme.typography.fontFamily,
    color: theme.palette.common.white,
  },
  logo: {
    height: 28,
    marginRight: 10
  },
  overviewHover: {
    '&:hover': {
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
    },
  },
  itemActiveItem: {
    color: theme.palette.secondary.main,
  },
});


class Navigator extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    location: PropTypes.object,
  }

  render() {
    const {
      classes,
      location,
      staticContext,
      open,
      onClose,
      ...other
    } = this.props;

    const navLinks = [
      /*{
        url: "/publications",
        label: "Publicationcs",
      },*/
      {
        url: "/contact",
        label: "Contact",
      },
    ]

    return (
      <Drawer
        variant="permanent"
        open={open}
        onClose={onClose}
        {...other}
      >
        <List disablePadding>
          <ListItem
            component={Link}
            to="/"
            className={classNames(
              classes.firebase,
              classes.item,
              classes.itemCategory)}
          >
            <img alt="logo" src={Logo} className={classes.logo} />
            Rafiki
          </ListItem>
          {navLinks.map((link, index) => (
            <ListItem
              key={index}
              component={Link}
              to={link.url}
              onClick={onClose}
              className={classNames(
                classes.item,
                classes.overviewHover,
                classes.itemCategory,
                location.pathname === link.url &&
                classes.itemActiveItem,
                classes.categoryHeader
              )}
            >
              <ListItemText
                classes={
                  location.pathname === link.url
                    ? {
                      primary: classes.categoryHeaderPrimaryActive
                    }
                    : {
                      primary: classes.categoryHeaderPrimary
                    }
                  }
              >
                {link.label}
              </ListItemText>
            </ListItem>
          ))}
        </List>
      </Drawer>
    );
  }
}


export default compose(
  withRouter,
  withStyles(styles)
)(Navigator) // This is Navgigator 
