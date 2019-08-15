import React from 'react';
import { connect } from "react-redux"
import { compose } from "redux"
import PropTypes from 'prop-types';
import { Switch, Route, Redirect } from "react-router-dom"

import { MuiThemeProvider, withStyles } from '@material-ui/core/styles';
import Hidden from '@material-ui/core/Hidden';

import Header from 'components/Console/ConsoleHeader/Header';
import Navigator from 'components/Console/ConsoleSideBar/Navigator';
import ConsoleTheme from "./ConsoleTheme"

// Datasets Component
import ListDataSet from "../Datasets/ListDataSet"
import UploadDataset from "../Datasets/UploadDataset"

// Trainjobs Component
import ListTrainJobs from "../Jobs/ListTrainJobs"
import CreateTrainJob from "../Jobs/CreateTrainJob"
import ListTrials from "../Jobs/ListTrials"
import TrialDetails from "../Jobs/TrialsDetails"

// Inference Jobs Component
import ApplicationDetails from "../Application/ApplicationDetails"
import ListApplication from "../Application/ListApplication"
import CreateInferenceJob from '../Application/CreateInferenceJob';

import InProgress from "../WorkInProgress/InProgress"

import * as actions from "./actions"

import LoadingBar from 'react-redux-loading-bar'


const drawerWidth = 256;

const styles = theme => ({
  root: {
    display: 'flex',
    minHeight: '100vh',
  },
  drawer: {
    [theme.breakpoints.up('md')]: {
      width: drawerWidth,
      flexShrink: 0,
    },
  },
  appContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
  },
})

class ConsoleAppFrame extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    mobileOpen: PropTypes.bool,
    headerTitle: PropTypes.string,
    handleDrawerToggle: PropTypes.func,
    // for StorageBar
    DBSize: PropTypes.string,
    StorageBarStatus: PropTypes.string
  }

  render() {
    const {
      authStatus,
      classes,
      handleDrawerToggle,
      headerTitle,
      mobileOpen,
    } = this.props;

    if (!authStatus) {
      return <Redirect to="/sign-in" />
    }

    return (
      <MuiThemeProvider theme={ConsoleTheme}>
        <LoadingBar
          // only display if the action took longer than updateTime to finish
          // default updateTime = 200ms
          updateTime={300}
          progressIncrease={10}
          style={{
            backgroundColor: '#fc6e43',
            height: 8,
            zIndex: 2000,
            position: "fixed",
            top: 0
          }}
        />
        <div className={classes.root}>
          <nav className={classes.drawer}>
            <Hidden mdUp implementation="js">
              <Navigator
                PaperProps={{ style: { width: drawerWidth } }}
                variant="temporary"
                open={mobileOpen}
                onClose={handleDrawerToggle}
              />
            </Hidden>
            <Hidden smDown implementation="css">
              <Navigator PaperProps={{ style: { width: drawerWidth } }} />
            </Hidden>
          </nav>
          <div className={classes.appContent}>
            <Header
              onDrawerToggle={handleDrawerToggle}
              title={headerTitle}
            />
            <Switch>
              <Route
                exact
                path='/console/datasets/list-dataset'
                component={ListDataSet}
              />
              <Route
                exact
                path='/console/datasets/upload-datasets'
                component={UploadDataset}
              />
              <Route
                exact
                path='/console/datasets/delete-dataset'
                component={InProgress}
              />
              <Route
                exact
                path='/console/jobs/list-train-jobs'
                component={ListTrainJobs}
              />
              <Route
                exact
                path='/console/jobs/create-train-job'
                component={CreateTrainJob}
              />
              <Route
                exact
                path='/console/jobs/trials/:appId/:app/:appVersion'
                component={ListTrials}
              />
              <Route
                exact
                path='/console/jobs/trials/:trialId'
                component={TrialDetails}
              />
              <Route
                exact
                path='/console/application/:appId/:app/:appVersion/create_inference_job'
                component={CreateInferenceJob}
              />
              <Route
                exact
                path="/console/application/list-applications"
                component={ListApplication}
              />
              <Route
                exact
                path='/console/application/running_job/:app/:appVersion'
                component={ApplicationDetails}
              />
            </Switch>
          </div>
        </div>
      </MuiThemeProvider>
    );
  }
}

const mapStateToProps = state => ({
  authStatus: !!state.Root.token,
  mobileOpen: state.ConsoleAppFrame.mobileOpen,
  headerTitle: state.ConsoleAppFrame.headerTitle,
})

const mapDispatchToProps = {
  handleDrawerToggle: actions.handleDrawerToggle
}


export default compose(
  connect(
    mapStateToProps,
    mapDispatchToProps
  ),
  withStyles(styles)
)(ConsoleAppFrame);
