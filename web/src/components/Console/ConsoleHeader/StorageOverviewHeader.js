import React from 'react';
import PropTypes from 'prop-types';
import AppBar from '@material-ui/core/AppBar';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import { withStyles } from '@material-ui/core/styles';


const styles = () => ({
  secondaryBar: {
    zIndex: 0,
    backgroundColor: "#eaeff1"
  },
});

function Header(props) {
  const { classes } = props;

  return (
    <React.Fragment>
      <AppBar
        component="div"
        className={classes.secondaryBar}
        position="static"
        elevation={0}
      >
        <Tabs value={0} textColor="inherit">
          <Tab textColor="inherit" label="Information" />
        </Tabs>
      </AppBar>
    </React.Fragment>
  );
}

Header.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(Header);


