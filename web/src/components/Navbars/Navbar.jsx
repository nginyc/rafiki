import React from "react";
import classNames from "classnames";
import PropTypes from "prop-types";
// @material-ui/core components
import { Link } from "react-router-dom";

import withStyles from "@material-ui/core/styles/withStyles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import Hidden from "@material-ui/core/Hidden";
// @material-ui/icons
import Menu from "@material-ui/icons/Menu";
// core components
import AdminNavbarLinks from "./AdminNavbarLinks.jsx";
import Button from "components/CustomButtons/Button.jsx";
import headerStyle from "assets/jss/material-dashboard-react/components/headerStyle.jsx";

function getUrlDirectory(path) { // url = '/admin/datasets/new' directory = '/admin/datasets'
  const directory = path.substring(0,path.lastIndexOf("/"));
  return directory
}

/* function getUrlBase(path) { // base = '/new'
  const base = path.substring(path.lastIndexOf("/"))
  return base
} */

function Header({ ...props }) {
  function makeBrands() {
    let brands = [];
    let pathName = props.location.pathname
    while(true) {
      props.routes.map((prop, key) => {
        if (prop.layout + prop.path === pathName) { // props.loation.pathname = "/admin/jobs/new"
          let brand = {}
          brand.name = prop.name;
          if (getUrlDirectory(pathName) === "/admin") {
            brand.href = prop.layout + prop.path
          } else {
            brand.href = undefined
          }
          brands.push(brand)
        }
        return null;
      });
      if ((pathName = getUrlDirectory(pathName)) === "") {
        break;
      }
    }
    return brands.reverse()
  }
  const { classes, color } = props;
  const brands = makeBrands().map((brand)=> {
    /* Here we create navbar brand, based on route name */
    return brand.href ? 
      (
        <Link to={brand.href} className={classes.link}>
          <Button color="transparent" className={classes.title}>
            {brand.name}
          </Button> 
        </Link> 
      ) : (
       <Button color="transparent" className={classes.invalid_title}>
          {brand.name}
      </Button> 
      )
    }
  )
  const appBarClasses = classNames({
    [" " + classes[color]]: color
  });
  return (
    <AppBar className={classes.appBar + appBarClasses}>
      <Toolbar className={classes.container}>
        <div className={classes.flex} >
          {brands}
        </div>
        <Hidden smDown implementation="css">
          { <AdminNavbarLinks onLogout = {props.onLogout} />}
        </Hidden>
        <Hidden mdUp implementation="css">
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={props.handleDrawerToggle}
          >
            <Menu />
          </IconButton>
        </Hidden> 
      </Toolbar>
    </AppBar>
  );
}

Header.propTypes = {
  classes: PropTypes.object.isRequired,
  color: PropTypes.oneOf(["primary", "info", "success", "warning", "danger"])
};

export default withStyles(headerStyle)(Header);
