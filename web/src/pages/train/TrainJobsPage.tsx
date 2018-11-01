import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { Typography, Paper, CircularProgress,
  Table, TableHead, TableCell, TableBody, TableRow, Icon, IconButton } from '@material-ui/core';
import { Pageview } from '@material-ui/icons';

import { AppUtils } from '../../App';
import { AppRoute } from '../../app/AppNavigator';
import { TrainJob } from '../../../client/RafikiClient';

interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
}

class TrainJobsPage extends React.Component<Props> {
  state: {
    trainJobs: TrainJob[] | null
  } = {
    trainJobs: null
  }

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError } } = this.props;
    const user = rafikiClient.getCurrentUser();
    try {
      const trainJobs = await rafikiClient.getTrainJobsByUser(user.id);
      this.setState({ trainJobs });
    } catch (error) {
      showError(error, 'Failed to retrieve train jobs');
    }
  }

  renderTrainJobs() {
    const { appUtils: { appNavigator } } = this.props;
    const { trainJobs } = this.state;

    return (
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>ID</TableCell>
            <TableCell>App</TableCell>
            <TableCell>App Version</TableCell>
            <TableCell>Task</TableCell>
            <TableCell>Budget</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Started At</TableCell>
            <TableCell>Completed At</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {trainJobs.map(x => {
            return (
              <TableRow key={x.id} hover>
                <TableCell>{x.id}</TableCell>
                <TableCell>{x.app}</TableCell>
                <TableCell>{x.app_version}</TableCell>
                <TableCell>{x.task}</TableCell>
                <TableCell>{x.budget_amount}</TableCell>
                <TableCell>{x.status}</TableCell>
                <TableCell>{x.datetime_started}</TableCell>
                <TableCell>{x.datetime_completed}</TableCell>
                <TableCell>
                  <IconButton onClick={() => {
                    const link = AppRoute.TRAIN_JOB_DETAIL
                      .replace(':app', x.app)
                      .replace(':appVersion', x.app_version);
                    appNavigator.goTo(link);
                  }}>
                    <Pageview /> 
                  </IconButton>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    )
  }

  render() {
    const { classes, appUtils } = this.props;
    const { trainJobs } = this.state;

    return (
      <main className={classes.main}>
          <Paper className={classes.jobsPaper}>
            {
              trainJobs &&
              this.renderTrainJobs()
            }
            {
              !trainJobs &&
              <CircularProgress />
            }
          </Paper>
          
      </main>
    );
  }
}

const styles: StyleRulesCallback = (theme) => ({
  main: {
  },
  jobsPaper: {
    overflowX: 'auto'
  }
});

export default withStyles(styles)(TrainJobsPage);