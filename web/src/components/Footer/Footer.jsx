import React from "react";
import PropTypes from "prop-types";
// @material-ui/core components
import withStyles from "@material-ui/core/styles/withStyles";
import ListItem from "@material-ui/core/ListItem";
import List from "@material-ui/core/List";
// core components
import footerStyle from "assets/jss/material-dashboard-react/components/footerStyle.jsx";

function Footer({ ...props }) {
  const { classes } = props;
  return (
    <footer className={classes.footer}>
      <div className={classes.container}>
        <div className={classes.left}>
          <List className={classes.list}>
            <ListItem className={classes.inlineBlock}>
              <a href="https://singa.incubator.apache.org" target="_blank" className={classes.block}>
                About Singa
              </a>
            </ListItem>
            <ListItem className={classes.inlineBlock}>
              <a href="https://nginyc.github.io/rafiki/docs/latest/" target="_blank" className={classes.block}>
                About Rafiki
              </a>
            </ListItem>
          </List>
        </div>
        <p className={classes.right}>
          <span>
            &copy; {1900 + new Date().getYear()}{" "} made with love by&nbsp;
            <a href="https://www.comp.nus.edu.sg/~dbsystem/#" target="_blank" className={classes.a}>
               DBSystem
            </a>
          
          </span>
        </p>
      </div>
    </footer>
  );
}

Footer.propTypes = {
  classes: PropTypes.object.isRequired
};

export default withStyles(footerStyle)(Footer);
