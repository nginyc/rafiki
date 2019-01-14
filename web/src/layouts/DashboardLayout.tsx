/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { AppBar, Toolbar, Typography, Drawer, List, Hidden,
  ListItem, ListItemIcon, ListItemText } from '@material-ui/core';
import IconButton from '@material-ui/core/IconButton';
import { Menu, Schedule, ExitToApp } from '@material-ui/icons';
import { Route, Switch, Redirect } from 'react-router-dom';

import { AppUtils } from '../App';
import { AppRoute } from '../app/AppNavigator';
import TrainJobsPage from '../pages/train/TrainJobsPage';
import TrainJobDetailPage from '../pages/train/TrainJobDetailPage';
import TrialDetailPage from '../pages/train/TrialDetailPage';

interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
}

class DashboardLayout extends React.Component<Props> {
  state = {
    isDrawerOpen: false
  }

  onLogout() {
    const { appUtils: { rafikiClient, appNavigator } } = this.props;
    rafikiClient.logout();
    appNavigator.goTo(AppRoute.LOGIN);
  }

  renderNavItems() {
    const { appUtils: { appNavigator }} = this.props;
    return (
      <List>
        <ListItem button onClick={() => appNavigator.goTo(AppRoute.TRAIN_JOBS)}>
          <ListItemIcon><Schedule /></ListItemIcon>
          <ListItemText primary="Train Jobs" />
        </ListItem>
        {/* <ListItem button onClick={() => appNavigator.goTo(AppRoute.INFERENCE_JOBS)}>
          <ListItemIcon><CloudUpload /></ListItemIcon>
          <ListItemText primary="Inference Jobs" />
        </ListItem> */}
      </List>
    );
  }

  renderNav() {
    const { isDrawerOpen } = this.state;
    const { classes } = this.props;

    return (
      <nav>
        <Hidden mdUp>
          <Drawer classes={{ paper: classes.navDrawer }} 
            variant="temporary" open={isDrawerOpen} onClose={() => this.setState({ isDrawerOpen: false })}>
            {this.renderNavItems()}
          </Drawer>
        </Hidden>
        <Hidden smDown>
          <Drawer classes={{ paper: classes.navDrawer }} variant="permanent">
            {this.renderNavItems()}
          </Drawer>
        </Hidden>
      </nav>
    );
  }

  renderBar() {
    const { classes } = this.props;
    const { isDrawerOpen } = this.state;

    return (
      <AppBar position="static">
        <Toolbar>
          <Hidden mdUp>
            <IconButton onClick={() => this.setState({ isDrawerOpen: !isDrawerOpen })} 
              color="inherit" aria-label="Menu">
                <Menu />
            </IconButton>
          </Hidden>
          <div className={classes.barMain}>
            <Typography variant="h6" color="inherit">
              Rafiki Admin Web
            </Typography>
          </div>
          <IconButton onClick={() => this.onLogout()} color="inherit" aria-label="Logout">
            <ExitToApp />
          </IconButton>
        </Toolbar>
      </AppBar>
    );
  }

  renderPage() {
    const { appUtils, classes } = this.props;

    return (
      <main className={classes.main}>
        <Switch>
          <Route exact path={AppRoute.TRAIN_JOBS} render={(props) => {
            return <TrainJobsPage appUtils={appUtils} />;
          }}/>
          <Route exact path={AppRoute.TRAIN_JOB_DETAIL} render={(props) => {
            const { app, appVersion } = props.match.params;
            return <TrainJobDetailPage app={app} appVersion={parseInt(appVersion)} appUtils={appUtils} />;
          }}/>
          <Route exact path={AppRoute.TRIAL_DETAIL} render={(props) => {
            const { trialId } = props.match.params;
            return <TrialDetailPage trialId={trialId} appUtils={appUtils} />;
          }}/>
          <Redirect to={AppRoute.TRAIN_JOBS} />
        </Switch>
      </main>  
    );
  }

  render() {
    const { classes } = this.props;

    return (
      <React.Fragment>
        <div className={classes.root}>
          <div>
            {this.renderNav()}
          </div>
          <div className={classes.mainBox}>
            {this.renderBar()}
            {this.renderPage()}
          </div>
        </div>
      </React.Fragment>
    );
  }
}

const styles: StyleRulesCallback = (theme) => ({
  root: {
  },
  mainBox: {
    [theme.breakpoints.up('md')]: {
      marginLeft: 240
    },
    display: 'flex',
    flexDirection: 'column',
    height: '100vh'
  },
  main: {
    overflow: 'auto',
    padding: theme.spacing.unit * 4
  },
  barMain: {
    flexGrow: 1
  },
  navDrawer: {
    width: 240
  }
});

export default withStyles(styles)(DashboardLayout);