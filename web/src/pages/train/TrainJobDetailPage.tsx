import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { Typography, Paper, CircularProgress,
  Table, TableHead, TableCell, TableBody, TableRow, Icon, IconButton } from '@material-ui/core';
import { Pageview } from '@material-ui/icons';

import { AppUtils } from '../../App';
import { AppRoute } from '../../app/AppNavigator';
import { Trial } from '../../../client/RafikiClient';

interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
  app: string;
  appVersion: number;
}

class TrainJobDetailPage extends React.Component<Props> {
  state: {
    trials: Trial[] | null
  } = {
    trials: null
  }

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError }, app, appVersion } = this.props;
    const user = rafikiClient.getCurrentUser();
    try {
      const trials = await rafikiClient.getTrialsOfTrainJob(app, appVersion);
      this.setState({ trials });
    } catch (error) {
      showError(error, 'Failed to retrieve trials for train job');
    }
  }

  renderTrials() {
    const { appUtils: { appNavigator } } = this.props;
    const { trials } = this.state;

    return (
      <Table padding="dense">
        <TableHead>
          <TableRow>
            <TableCell padding="none"></TableCell>
            <TableCell>ID</TableCell>
            <TableCell>Model</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Score</TableCell>
            <TableCell>Started At</TableCell>
            <TableCell>Stopped At</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {trials.map(x => {
            return (
              <TableRow key={x.id} hover>
                <TableCell padding="none">
                  <IconButton onClick={() => {
                    const link = AppRoute.TRIAL_DETAIL
                      .replace(':trialId', x.id)
                    appNavigator.goTo(link);
                  }}>
                    <Pageview /> 
                  </IconButton>
                </TableCell>
                <TableCell>{x.id}</TableCell>
                <TableCell>{x.model_name}</TableCell>
                <TableCell>{x.status}</TableCell>
                <TableCell>{x.score}</TableCell>
                <TableCell>{x.datetime_started}</TableCell>
                <TableCell>{x.datetime_stopped}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    )
  }

  render() {
    const { classes, appUtils, app, appVersion } = this.props;
    const { trials } = this.state;

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h2">
          Train Job 
          <span className={classes.headerSub}>{`(${app} V${appVersion})`}</span>
        </Typography>
        <Typography gutterBottom variant="h3">Trials</Typography>
        <Paper className={classes.trialsPaper}>
          {
            trials &&
            this.renderTrials()
          }
          {
            !trials &&
            <CircularProgress />
          }
        </Paper>
      </React.Fragment>
    );
  }
}

const styles: StyleRulesCallback = (theme) => ({
  headerSub: {
    fontSize: theme.typography.h4.fontSize,
    margin: theme.spacing.unit * 2
  },
  trialsPaper: {
    overflowX: 'auto'
  }
});

export default withStyles(styles)(TrainJobDetailPage);