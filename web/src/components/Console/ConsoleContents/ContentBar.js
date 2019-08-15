import React from 'react';
import PropTypes from 'prop-types';

import AppBar from '@material-ui/core/AppBar';
import { withStyles } from '@material-ui/core/styles';


const styles = () => ({
  ContentBar: {
    borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
  }
})

class ContentBar extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    children: PropTypes.node,
  }

  render() {
    const { classes, children } = this.props;
    return (
      <React.Fragment>
        <AppBar
          className={classes.ContentBar}
          position="static"
          color="default"
          elevation={0}
        >
          {children}
        </AppBar>
      </React.Fragment>

    )
  }
}

export default withStyles(styles)(ContentBar)