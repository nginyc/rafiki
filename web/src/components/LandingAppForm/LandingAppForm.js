import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import LayoutBody from '../LandingComponents/LayoutBody';
import Paper from '../LandingComponents/Paper';

const styles = theme => ({
  root: {
    display: 'flex',
    backgroundRepeat: 'no-repeat',
  },
  paper: {
    padding: `${theme.spacing(1) * 4}px ${theme.spacing(1) * 3}px`,
    [theme.breakpoints.up('md')]: {
      padding: `${theme.spacing(1) * 10}px ${theme.spacing(1) * 8}px`,
    },
  },
});

function AppForm(props) {
  const { children, classes } = props;

  return (
    <div className={classes.root}>
      <LayoutBody margin marginBottom width="small">
        <Paper className={classes.paper}>{children}</Paper>
      </LayoutBody>
    </div>
  );
}

AppForm.propTypes = {
  children: PropTypes.node.isRequired,
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(AppForm);
