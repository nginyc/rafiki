import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { AppBar, Toolbar, Typography, Drawer, List, Hidden,
  ListItem, ListItemIcon, ListItemText } from '@material-ui/core';
import IconButton from '@material-ui/core/IconButton';
import { Menu, Schedule, CloudUpload } from '@material-ui/icons';
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
          <Typography variant="h6" color="inherit">
            Rafiki Web Admin
          </Typography>
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
  navDrawer: {
    width: 240
  }
});

export default withStyles(styles)(DashboardLayout);