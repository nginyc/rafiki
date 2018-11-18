import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { Typography, Paper, CircularProgress,
  Table, TableHead, TableCell, TableBody, TableRow, Icon, IconButton } from '@material-ui/core';
import { Pageview } from '@material-ui/icons';
import * as moment from 'moment';

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
    const { appUtils: { appNavigator }, classes } = this.props;
    const { trainJobs } = this.state;

    return (
      <Paper className={classes.jobsPaper}>
        <Table padding="dense">
          <TableHead>
            <TableRow>
              <TableCell padding="none"></TableCell>
              <TableCell>ID</TableCell>
              <TableCell>App</TableCell>
              <TableCell>App Version</TableCell>
              <TableCell>Task</TableCell>
              <TableCell>Budget</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Started At</TableCell>
              <TableCell>Completed At</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {trainJobs.map(x => {
              return (
                <TableRow key={x.id} hover>
                  <TableCell padding="none">
                    <IconButton onClick={() => {
                      const link = AppRoute.TRAIN_JOB_DETAIL
                        .replace(':app', x.app)
                        .replace(':appVersion', x.app_version);
                      appNavigator.goTo(link);
                    }}>
                      <Pageview /> 
                    </IconButton>
                  </TableCell>
                  <TableCell>{x.id}</TableCell>
                  <TableCell>{x.app}</TableCell>
                  <TableCell>{x.app_version}</TableCell>
                  <TableCell>{x.task}</TableCell>
                  <TableCell>{JSON.stringify(x.budget)}</TableCell>
                  <TableCell>{x.status}</TableCell>
                  <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                  <TableCell>{x.datetime_completed ? moment(x.datetime_completed).fromNow() : '-'}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>
    )
  }

  render() {
    const { classes, appUtils } = this.props;
    const { trainJobs } = this.state;

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h2">Your Train Jobs</Typography>
        {
          trainJobs &&
          this.renderTrainJobs()
        }
        {
          !trainJobs &&
          <CircularProgress />
        }
      </React.Fragment>
    );
  }
}

const styles: StyleRulesCallback = (theme) => ({
  jobsPaper: {
    overflowX: 'auto'
  }
});

export default withStyles(styles)(TrainJobsPage);