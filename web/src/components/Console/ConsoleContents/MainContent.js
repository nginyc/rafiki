import React from 'react';
import PropTypes from 'prop-types';
import { compose } from "redux";

import Paper from '@material-ui/core/Paper';
import { withStyles } from '@material-ui/core/styles';


const styles = theme => ({
  mainContent: {
    flex: 1,
    padding: '48px 36px 0',
    background: '#eaeff1', // light grey
  },
  paper: {
    maxWidth: 1280,
    margin: 'auto',
    overflow: 'hidden',
    marginBottom: 20,
    position: "relative",
    paddingBottom: 20
  },
  fab: {
    position: 'absolute',
    bottom: theme.spacing(1) * 3,
    right: theme.spacing(1) * 3,
    zIndex: 10
  },
  extendedIcon: {
    marginRight: theme.spacing(1),
  },
})

class MainContent extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    children: PropTypes.node,
    location: PropTypes.object,
  }

  render() {
    const { classes, children } = this.props;
    return (
      <main className={classes.mainContent}>
        <Paper className={classes.paper}>
          {children}
          {/* {location.pathname === "/console/datasets/list-dataset"
            ? (
              <Zoom in={true} unmountOnExit>
                <Fab
                  className={classes.fab}
                  color="primary"
                  component={Link}
                  to="/console/datasets/upload-datasets?addNewDS=TRUE"
                >
                  <AddIcon />
                </Fab>
              </Zoom>
            )
            : (
              <Zoom in={true} unmountOnExit>
                <Fab
                  variant="extended"
                  className={classes.fab}
                  color="primary"
                  component={Link}
                  to="/console/datasets/list-dataset"
                >
                  <ListDSIcon className={classes.extendedIcon} />
                  List Dataset
                </Fab>
              </Zoom>
            )
          } */}
        </Paper>
      </main>
    )
  }
}

export default compose(
  withStyles(styles)
)(MainContent)
